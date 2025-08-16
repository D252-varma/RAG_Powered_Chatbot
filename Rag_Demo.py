import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os
from dotenv import load_dotenv
from datetime import datetime
import asyncio

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load .env and API key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Ensure event loop for async
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:", layout="wide")

# Inject CSS for chat layout, fixed input box, and fixed CSV download button
st.markdown("""
<style>
.chat-container {
    width: 100%;
    max-width: 700px;
    margin: 0 auto;
    padding-bottom: 90px;
    display: flex;
    flex-direction: column;
}
.message.user {
    background-color: #2b313e;
    color: white;
    padding: 1rem;
    border-radius: 1rem 1rem 0 1rem;
    margin: 1rem 1rem 0 0;
    align-self: flex-end;
    max-width: 80%;
    text-align: right;
    white-space: pre-wrap;
}
.message.bot {
    background-color: #475063;
    color: white;
    padding: 1rem;
    border-radius: 1rem 1rem 1rem 0;
    margin: 1rem 0 0 1rem;
    align-self: flex-start;
    max-width: 80%;
    text-align: left;
    white-space: pre-wrap;
}
.fixed-input {
    position: fixed;
    bottom: 15px;
    left: 370px;
    right: 20px;
    padding: 0.5em;
    background: #222831;
    z-index: 9999;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.fixed-input input[type="text"] {
    flex-grow: 1;
    padding: 0.75rem;
    border-radius: 8px;
    border: none;
    font-size: 1rem;
}
.fixed-input button {
    background-color: #4CAF50;
    border: none;
    color: white;
    padding: 0.75rem 1rem;
    font-size: 1.2rem;
    border-radius: 8px;
    cursor: pointer;
}
.download-btn-fixed {
    position: fixed;
    bottom: 15px;
    right: 50px;
    z-index: 9999;
}
</style>
""", unsafe_allow_html=True)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def run_chat(user_question, api_key, pdf_docs):
    text_chunks = get_text_chunks(get_pdf_text(pdf_docs))
    get_vector_store(text_chunks, api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)
    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response['output_text']

def main():
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'pdf_docs' not in st.session_state:
        st.session_state.pdf_docs = None

    # Sidebar: upload PDFs and submit
    st.sidebar.header("Upload PDFs")
    pdf_docs = st.sidebar.file_uploader("Upload PDF Files", accept_multiple_files=True)
    if st.sidebar.button("Submit & Process"):
        if pdf_docs:
            st.session_state.pdf_docs = pdf_docs
            st.sidebar.success("PDFs processed!")
        else:
            st.sidebar.warning("Please upload PDF files.")

    # Chat area
    st.title("RAG Powered Chatbot 🦾")
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for entry in st.session_state.conversation_history:
        user_q, bot_a, timestamp = entry
        st.markdown(f'<div class="message user">{user_q}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="message bot">{bot_a}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Fixed input box with form for submitting chat
    with st.container():
        st.markdown('<div class="fixed-input">', unsafe_allow_html=True)
        with st.form("chat_form", clear_on_submit=True):
            cols = st.columns([10, 1])
            user_question = cols[0].text_input("Ask a Question...", key="user_question", label_visibility="collapsed", placeholder="Type your question and press Enter or click →")
            submit_clicked = cols[1].form_submit_button("→")
        st.markdown('</div>', unsafe_allow_html=True)

    if user_question and submit_clicked:
        if st.session_state.pdf_docs:
            bot_response = run_chat(user_question, API_KEY, st.session_state.pdf_docs)
            st.session_state.conversation_history.append((user_question, bot_response, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            st.rerun()
        else:
            st.warning("Please upload PDF files and press Submit & Process before chatting.")

    # Fixed download button at bottom right
    if len(st.session_state.conversation_history) > 0:
        df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Answer", "Timestamp"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.markdown(
            f'<div class="download-btn-fixed">'
            f'<a href="file/csv;base64,{b64}" download="conversation_history.csv">'
            f'<button style="padding:0.8em 1.5em;background:#4CAF50;color:#fff;border:none;border-radius:8px;cursor:pointer;">Download chat history as CSV</button></a></div>',
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
