# 📄 RAG PDF Question Answering App

A Retrieval-Augmented Generation (RAG) application built using LangChain, Streamlit, OpenAI, FAISS, and LangSmith.

This application allows users to upload a PDF document, automatically process its contents into embeddings, store them in a FAISS vector database, and ask natural language questions about the document.

---

## 🚀 Features

- Upload PDF documents through a Streamlit web interface
- Extract and process PDF content
- Split documents into chunks using LangChain Text Splitters
- Generate embeddings using Hugging Face Sentence Transformers
- Store embeddings in a local FAISS Vector Database
- Retrieve relevant document chunks based on user questions
- Generate context-aware answers using OpenAI GPT models
- LangSmith tracing for observability and debugging
- Modern Streamlit UI with Yahoo-inspired purple theme
- Local FAISS database persistence to avoid rebuilding on every query

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- OpenAI
- Hugging Face Embeddings
- FAISS Vector Store
- LangSmith
- PyPDF

---

## 📂 Project Structure

```text
RAG-App/
│
├── app.py
├── .env
├── .gitignore
├── requirements.txt
├── README.md
│
└── macro_faiss_db/
```

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/jnehem-blip/RAG-App.git
cd RAG-App
```

### 2. Create Virtual Environment

```bash
python -m venv myvenv
```

Activate:

**Windows**

```bash
myvenv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
OPENAI_API_KEY=your_openai_api_key

LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=RAG-App
```

⚠️ Never commit your `.env` file to GitHub.

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will launch in your browser.

---

## 🔄 How It Works

1. Upload a PDF document.
2. PDF content is extracted using PyPDFLoader.
3. Text is split into chunks using RecursiveCharacterTextSplitter.
4. Chunks are converted into embeddings using Hugging Face.
5. Embeddings are stored in a FAISS vector database.
6. User asks a question.
7. Retriever fetches relevant chunks.
8. OpenAI generates an answer grounded in the document content.
9. LangSmith captures traces for debugging and monitoring.

---

## 📊 LangSmith Tracing

This project includes LangSmith tracing support.

LangSmith helps visualize:

- User queries
- Retrieved chunks
- LLM prompts
- Responses
- Execution timing

---

## 🎯 Learning Objectives

This project demonstrates:

- Data Ingestion
- Text Splitting
- Embeddings
- Vector Databases
- Retrievers
- RetrievalQA Chains
- Streamlit Frontend Development
- OpenAI Integration
- LangSmith Observability

---

## 👨‍💻 Author

**Joseph Nehemiah Jones**

Ad Operations Specialist @ Yahoo

Currently learning:
- Python
- LangChain
- Streamlit
- FastAPI
- RAG Applications
- Generative AI Engineering

GitHub:
https://github.com/jnehem-blip

---

## 📜 License

This project is created for educational and learning purposes.
