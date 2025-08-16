# RAG-Powered PDF Chatbot

A **Retrieval-Augmented Generation (RAG)** powered chatbot that lets users **upload PDFs** and **ask questions** based on their content.  
It extracts text from PDFs, embeds them using **Google Gemini Embeddings (`embedding-001`)**, stores them in a **FAISS vector database**, and retrieves context to provide accurate answers with **LangChain** and **Google Generative AI**.  

Built with **Streamlit** for an interactive UI. 🚀  

---

## ✨ Features

- 📂 Upload one or more **PDF documents**  
- 🧩 Automatic **text extraction & chunking** using LangChain's `RecursiveCharacterTextSplitter`  
- 🔍 **Vector search** with FAISS for efficient retrieval  
- 🤖 **RAG-based Q&A** powered by **Google Generative AI (Gemini)**  
- ⚡ Ask natural language questions, get **precise answers from your PDFs**  
- 🎨 Simple and interactive **Streamlit UI**  

---

## 🛠️ Tech Stack

- **[Streamlit](https://streamlit.io/)** – Web interface  
- **[PyPDF2](https://pypi.org/project/pypdf2/)** – PDF parsing  
- **[LangChain](https://www.langchain.com/)** – Framework for RAG pipelines  
- **[FAISS](https://faiss.ai/)** – Vector database  
- **[Google Generative AI](https://ai.google/)** – Embeddings & Chat model  
- **dotenv** – Environment variable management  

---

## 📂 Project Structure
📦 rag-pdf-chatbot
├── app.py                 # Main Streamlit app
├── requirements.txt       # Dependencies
├── .env                   # API keys and environment variables
└── README.md              # Project documentation

⸻

🧩 How It Works
	1.	Upload PDF → Extracts text with PyPDF2
	2.	Chunking → Splits text into smaller segments via RecursiveCharacterTextSplitter
	3.	Embedding → Converts text chunks into vector embeddings with Gemini embedding-001
	4.	Vector Store → Stores embeddings in FAISS for fast similarity search
	5.	Query → User asks a question → Retrieve top matches
	6.	RAG → Context + Question passed to Gemini Chat model → Generates final answer

⸻

📌 Example Workflow
	•	Upload a research paper PDF
	•	Ask: “What are the key findings in this paper?”
	•	The chatbot retrieves relevant sections and generates a summarized response

⸻

📜 Requirements
	•	Python 3.9+
	•	Google API Key (for Generative AI)

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/rag-pdf-chatbot.git
   cd rag-pdf-chatbot
2.Create a virtual environment
   python -m venv venv
  source venv/bin/activate   # Mac/Linux
  venv\Scripts\activate      # Windows
3.Install dependencies
  pip install -r requirements.txt
4.Set up environment variables
Create a .env file in the project root with:
  GOOGLE_API_KEY=your_google_api_key

Run the Streamlit app:
  streamlit run app.py
1.	Upload your PDF(s)
2.	Ask a question in the chat box
3.	The bot retrieves relevant chunks from the PDF using FAISS and answers using Google Gemini

## WEB INTERFACE

<img width="1440" height="858" alt="Screenshot 2025-08-15 at 11 05 00 PM" src="https://github.com/user-attachments/assets/6dd2294f-5580-4515-932a-39297ee5ba6a" />
=============================================================================================
<img width="1440" height="851" alt="Screenshot 2025-08-15 at 11 05 23 PM" src="https://github.com/user-attachments/assets/165374b1-993d-4e4e-98b6-6cd65d7cdd9c" />
=============================================================================================
<img width="1440" height="859" alt="Screenshot 2025-08-15 at 11 05 51 PM" src="https://github.com/user-attachments/assets/c28417f8-a553-43ff-b211-099a9a65fa4a" />
