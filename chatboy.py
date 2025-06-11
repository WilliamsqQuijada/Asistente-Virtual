import streamlit as st
from streamlit_chat import message
from langchain_openai import ChatOpenAI
import time
from langdetect import detect
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import tempfile
import os
import speech_recognition as sr
from dotenv import load_dotenv  

# Cargar variables de entorno
load_dotenv()

# Obtener la API Key desde el archivo .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

# ====== Estado inicial ======
if "page" not in st.session_state:
    st.session_state.page = "home"
if "chats_por_perfil" not in st.session_state:
    st.session_state.chats_por_perfil = {
        "Asistente General": {"Chat 1": []},
        "Asistente de TI": {"Chat 1": []}
    }
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Chat 1"
if "perfil" not in st.session_state:
    st.session_state.perfil = "Asistente General"
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False
if "transcribed_voice" not in st.session_state:
    st.session_state.transcribed_voice = None

# ====== CSS ======
st.markdown("""
<style>
button[kind="secondary"] {
    background-color: transparent !important;
    color: white !important;
    border: none !important;
    padding: 0 !important;
    font-size: 16px !important;
    text-align: left !important;
}
div[data-testid="stSidebar"] {
    background-color: #1e1e1e;
}
</style>
""", unsafe_allow_html=True)

# ====== Funciones de Páginas ======
def home():
    st.title("\U0001F44B Bienvenido a AuditBot tu AI de confianza")
    st.markdown("Selecciona con qué perfil quieres interactuar:")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("\U0001F9E0 Asistente General"):
            st.session_state.perfil = "Asistente General"
            st.session_state.page = "asistente"
    with col2:
        if st.button("\U0001F6E0️ Asistente de TI"):
            st.session_state.perfil = "Asistente de TI"
            st.session_state.page = "auditor"

def sidebar_config():
    with st.sidebar:
        if st.session_state.perfil:
            st.markdown(f"### Perfil actual: **{st.session_state.perfil}**")
            st.markdown("### \U0001F4C4 Subir archivo")

            if st.button("\U0001F4CE Cargar archivo"):
                st.session_state.show_uploader = not st.session_state.show_uploader

            uploaded_file = None
            if st.session_state.show_uploader:
                uploaded_file = st.file_uploader("Selecciona un archivo (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])

            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(tmp_path)
                elif uploaded_file.name.endswith(".txt"):
                    loader = TextLoader(tmp_path)
                elif uploaded_file.name.endswith(".docx"):
                    loader = Docx2txtLoader(tmp_path)

                docs = loader.load()

                if not docs:
                    st.warning(f"No se pudo extraer texto del archivo {uploaded_file.name}")
                else:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    split_docs = splitter.split_documents(docs)

                    if len(split_docs) == 0:
                        st.warning(f"El archivo {uploaded_file.name} no contiene suficiente texto para procesar.")
                    else:
                        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)  # 👈 ya está usando variable segura
                        db = FAISS.from_documents(split_docs, embeddings)
                        retriever = db.as_retriever()

                        st.session_state.qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            retriever=retriever,
                            chain_type="stuff"
                        )
                        st.success(f"✅ Archivo '{uploaded_file.name}' cargado correctamente.")

            # Grabar voz
            st.markdown("### 🎙️ Entrada por voz")
            if st.button("🎙️ Grabar voz"):
                recognizer = sr.Recognizer()
                mic = sr.Microphone()
                with mic as source:
                    audio = recognizer.listen(source)

                try:
                    texto = recognizer.recognize_google(audio, language="es-ES")
                    st.session_state.transcribed_voice = texto
                except (sr.UnknownValueError, sr.RequestError):
                    st.session_state.transcribed_voice = None

            # Chats
            st.markdown("### 💬 Tus chats")
            chats_del_perfil = st.session_state.chats_por_perfil.get(st.session_state.perfil, {})
            chat_names = list(chats_del_perfil.keys())

            for nombre_chat in chat_names:
                mensajes = chats_del_perfil[nombre_chat]
                if mensajes:
                    resumen = mensajes[0]["content"][:25] + "..." if len(mensajes[0]["content"]) > 25 else mensajes[0]["content"]
                    nombre_visible = resumen
                else:
                    nombre_visible = nombre_chat

                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    if st.button(nombre_visible, key=f"chat_{nombre_chat}"):
                        st.session_state.current_chat = nombre_chat
                with col2:
                    if st.button("❌", key=f"delete_{nombre_chat}"):
                        del st.session_state.chats_por_perfil[st.session_state.perfil][nombre_chat]
                        nuevos_chats = list(st.session_state.chats_por_perfil[st.session_state.perfil].keys())
                        st.session_state.current_chat = nuevos_chats[0] if nuevos_chats else None
                        st.rerun()

            if st.button("➕ Nuevo chat"):
                nuevo_nombre = f"Chat {len(chats_del_perfil) + 1}"
                st.session_state.chats_por_perfil[st.session_state.perfil][nuevo_nombre] = []
                st.session_state.current_chat = nuevo_nombre

        if st.button("🔙 Salir"):
            st.session_state.page = "home"
            st.session_state.perfil = None
            st.session_state.current_chat = "Chat 1"

def get_system_message(language, perfil):
    if perfil == "Asistente de TI":
        return (
            "Eres un técnico especializado en inventario de TI. "
            "Puedes registrar, consultar y auditar hardware, software y equipos de red. "
            "Usa lenguaje técnico y estructurado. Si es útil, responde en formato tabla."
        )
    else:
        if language == "es":
            return "Eres un asistente experto llamado AuditBot. Siempre responde como un humano."
        elif language == "en":
            return "You are an expert assistant named AuditBot. Always respond like a human."
        else:
            return "You are an expert assistant named AuditBot. Always respond like a human."

def chat_page():
    sidebar_config()

    perfil = st.session_state.perfil
    chats_del_perfil = st.session_state.chats_por_perfil[perfil]

    if not st.session_state.current_chat or st.session_state.current_chat not in chats_del_perfil:
        if chats_del_perfil:
            st.session_state.current_chat = list(chats_del_perfil.keys())[0]
        else:
            nuevo_chat = "Chat 1"
            st.session_state.chats_por_perfil[perfil][nuevo_chat] = []
            st.session_state.current_chat = nuevo_chat

    chat_history = chats_del_perfil[st.session_state.current_chat]

    st.markdown(f"# {('🧠' if perfil == 'Asistente General' else '🛠️')} AuditBot - {perfil}")

    for i, msg in enumerate(chat_history):
        is_user = msg["role"] == "user"
        message(msg["content"], is_user=is_user, key=f"msg_{i}")

    prompt = None
    if st.session_state.transcribed_voice:
        prompt = st.session_state.transcribed_voice
        st.session_state.transcribed_voice = None
    else:
        prompt = st.chat_input("¿En qué te puedo ayudar ahora?")

    if prompt:
        chat_history.append({"role": "user", "content": prompt})
        message(prompt, is_user=True, key=f"user_{len(chat_history)}")

        language = detect(prompt)

        with st.spinner("✍️ Escribiendo..."):
            system_message = get_system_message(language, perfil)
            messages = [("system", system_message)] + [(m["role"], m["content"]) for m in chat_history if m["role"] in ["user", "assistant"]]

            if st.session_state.qa_chain:
                try:
                    respuesta_con_contexto = st.session_state.qa_chain.invoke(prompt)
                    respuesta_con_contexto = respuesta_con_contexto.get("result", respuesta_con_contexto)
                except:
                    respuesta_con_contexto = None
            else:
                respuesta_con_contexto = None

            if respuesta_con_contexto:
                response = respuesta_con_contexto
            else:
                messages.append(("human", prompt))
                response = llm.invoke(messages).content

            time.sleep(1.2)
            chat_history.append({"role": "assistant", "content": response})
            message(response, is_user=False, key=f"assistant_{len(chat_history)}")

# ====== Ruteo de páginas ======
if st.session_state.page == "home":
    home()
elif st.session_state.page == "asistente":
    chat_page()
elif st.session_state.page == "auditor":
    chat_page()
