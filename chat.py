from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import google.generativeai as genai
from googletrans import Translator
import speech_recognition as sr
from pydub import AudioSegment
import mysql.connector
from datetime import datetime
import traceback

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

PDF_PATH = "loi.pdf"
FAISS_INDEX_PATH = "faiss_index"

# Configuration de la base de données
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = ""
DB_NAME = "todos"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            print(f"Error reading PDF: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)

if not os.path.exists(FAISS_INDEX_PATH):
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF file not found at {PDF_PATH}")
    raw_text = get_pdf_text([PDF_PATH])
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
try:
    new_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()
except Exception as e:
    raise RuntimeError(f"Error loading FAISS index: {e}")

model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0.3)
prompt = PromptTemplate.from_template("""
    Réponds à la question suivante en français, en utilisant le contexte fourni.
    Context: {context}
    Question: {question}
    Answer:
""")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
translator = Translator()
recognizer = sr.Recognizer()

def insert_into_db(question, response):
    try:
        connection = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)
        cursor = connection.cursor()
        query = "INSERT INTO messages (message, response, timestamp) VALUES (%s, %s, %s)"
        timestamp = datetime.now()
        cursor.execute(query, (question, response, timestamp))
        connection.commit()
        cursor.close()
        connection.close()
    except Exception as e:
        print(f"Error inserting into database: {e}")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_question = data.get('question')

    if not user_question:
        return jsonify({'error': 'Question is required'}), 400

    try:
        response = chain.invoke(user_question)

        try:
            translation = translator.translate(response, dest='fr')
            response = translation.text
        except Exception as translate_error:
            print(f"Translation error: {translate_error}")

        formatted_response = response.replace('\n', '<br>').replace('"', "'")
        insert_into_db(user_question, formatted_response)
        return jsonify({'response': formatted_response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/voice_chat', methods=['POST'])
def voice_chat():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    temp_audio_path = "temp_audio.wav"
    original_filename = audio_file.filename

    try:
        if original_filename.endswith(('.webm', '.wav', '.mp3')):
            try:
                if original_filename.endswith('.mp3'):
                    audio = AudioSegment.from_mp3(audio_file)
                else:
                    audio = AudioSegment.from_file(audio_file)
                audio.export(temp_audio_path, format="wav")
            except Exception as e:
                return jsonify({'error': f"Erreur lors de la conversion audio: {e}"}), 500
        else:
            return jsonify({'error': "Format audio non supporté"}), 400

        try:
            with sr.AudioFile(temp_audio_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language="fr-FR")
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': f"Erreur lors de la lecture du fichier WAV: {e}"}), 500

        os.remove(temp_audio_path)

        response = chain.invoke(text)

        try:
            translation = translator.translate(response, dest='fr')
            response = translation.text
        except Exception as translate_error:
            print(f"Translation error: {translate_error}")

        formatted_response = response.replace('\n', '<br>').replace('"', "'")
        insert_into_db(text, formatted_response)
        return jsonify({'response': formatted_response})

    except sr.UnknownValueError:
        return jsonify({'error': 'Could not understand audio'}), 400
    except sr.RequestError as e:
        return jsonify({'error': f'Speech recognition request failed; {e}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)