import os
import tempfile
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# ============================================
# LOAD ENVIRONMENT VARIABLES
# ============================================

load_dotenv()

print("LangSmith Project:", os.getenv("LANGSMITH_PROJECT"))

DB_PATH = "macro_faiss_db"

# ============================================
# STREAMLIT CONFIG
# ============================================

st.set_page_config(
    page_title="Yahoo Macro Assistant",
    page_icon="🚀",
    layout="wide"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>

.main {
    background-color: #FAF8FF;
}

.header {
    background: linear-gradient(90deg,#6001D2,#7B2CF5);
    padding: 20px;
    border-radius: 15px;
    color: white;
}

.answer-box {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border-left: 8px solid #6001D2;
}

.source-box {
    background-color: #F3EEFF;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================

st.markdown("""
<div class='header'>
<h1>🚀 Yahoo 3P Macro Assistant</h1>
<p>Search approved macros using LangChain RAG</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:

    st.header("📘 Technologies Used")

    st.success("LangChain + LangSmith")

    st.write("""
    ✅ PDF Loader

    ✅ Text Splitter

    ✅ HuggingFace Embeddings

    ✅ FAISS Vector Store

    ✅ Retriever

    ✅ GPT-4.1 Mini

    ✅ LangSmith Tracing
    """)

# ============================================
# LOAD EMBEDDINGS
# ============================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# ============================================
# BUILD VECTOR DATABASE
# ============================================

def build_vector_database(uploaded_pdf):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_pdf:

        temp_pdf.write(uploaded_pdf.read())
        pdf_path = temp_pdf.name

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vector_store.save_local(DB_PATH)

    return vector_store

# ============================================
# LOAD VECTOR DATABASE
# ============================================

def load_vector_database():

    if os.path.exists(DB_PATH):

        return FAISS.load_local(
            DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    return None

# ============================================
# PDF UPLOAD
# ============================================

st.subheader("📄 Upload Approved Macro Guide")

uploaded_file = st.file_uploader(
    "Upload Macro Guide PDF",
    type=["pdf"]
)

col1, col2 = st.columns(2)

with col1:

    if uploaded_file:

        if st.button("⚙️ Build Database"):

            with st.spinner(
                "Creating embeddings and vector database..."
            ):

                build_vector_database(uploaded_file)

            st.success(
                "Database created and saved locally."
            )

with col2:

    if st.button("📂 Load Existing Database"):

        if os.path.exists(DB_PATH):

            st.success(
                "Existing FAISS database loaded."
            )

        else:

            st.warning(
                "No saved database found."
            )

st.divider()

# ============================================
# LOAD VECTOR STORE
# ============================================

vector_store = load_vector_database()

if vector_store:

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    st.subheader("🔍 Search Approved Macros")

    question = st.text_input(
        "Example: What is the approved click macro?"
    )

    if st.button("Search Macro Guide"):

        if question.strip():

            with st.spinner(
                "Searching approved macros..."
            ):

                response = qa_chain.invoke(
                    {"query": question}
                )

            answer = response["result"]

            source_docs = response["source_documents"]

            st.subheader("💡 Answer")

            st.markdown(
                f"""
                <div class='answer-box'>
                {answer}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.subheader("📄 Source References")

            for idx, doc in enumerate(
                source_docs,
                start=1
            ):

                page = doc.metadata.get(
                    "page",
                    "Unknown"
                )

                st.markdown(
                    f"""
                    <div class='source-box'>
                    <b>Source {idx}</b><br>
                    <b>Page:</b> {page}<br><br>
                    {doc.page_content[:800]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

else:

    st.info(
        "Upload a PDF and build the vector database first."
    )

# ============================================
# FOOTER
# ============================================

st.divider()

st.caption(
    "Built with Streamlit • LangChain • LangSmith • HuggingFace • FAISS • OpenAI"
)