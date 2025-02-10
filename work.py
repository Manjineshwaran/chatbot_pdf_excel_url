import streamlit as st
import os
import warnings
from typing import List
from dotenv import load_dotenv
import pickle
import io
import requests
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from bs4 import BeautifulSoup

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')

def load_pdf(file_path: str) -> List[Document]:
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    except FileNotFoundError:
        st.error(f"Error: File {file_path} not found.")
        return []
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return []

def load_url(url: str) -> List[Document]:
    try:
        response = requests.get(url)
        response.raise_for_status()  
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.get_text()

        if not content.strip():
            st.error(f"Error: The content from the URL {url} is empty.")
            return []

        document = Document(page_content=content, metadata={"source": url})
        return [document]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL content: {e}")
        return []
    except Exception as e:
        st.error(f"Error loading content from URL {url}: {e}")
        return []

def load_excel(file: io.BytesIO) -> List[Document]:
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
        else:
            st.error(f"Unsupported file format: {file.name}")
            return []

        content = df.to_string(index=False)  
        document = Document(page_content=content, metadata={"source": file.name})
        return [document]
    except Exception as e:
        st.error(f"Error processing Excel/CSV: {e}")
        return []

def split_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

def create_vector_store(documents: List[Document], embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        vector_store = FAISS.from_documents(documents, embeddings)

        with open("faiss_vector_store.pkl", "wb") as f:
            pickle.dump(vector_store, f)

        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def load_vector_store():
    try:
        with open("faiss_vector_store.pkl", "rb") as f:
            vector_store = pickle.load(f)
        return vector_store
    except FileNotFoundError:
        return None

def create_qa_chain(vector_store):
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_retries=2
        )
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5} 
        )
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    except Exception as e:
        st.error(f"Error creating QA chain: {e}")
        return None

class DocumentProcessor:
    def __init__(self):
        self.vector_store = load_vector_store()
        self.qa_chain = None
        self.processed_files = []
    
    def process_documents(self, files: List[io.BytesIO], file_type: str, url: str = ""):
        new_files = [f for f in files if f.name not in self.processed_files]
        
        if not new_files and not url:
            st.info("No new documents to process.")
            return "No new documents to process."
        
        all_documents = []
        if url:
            documents = load_url(url)
            if documents:
                all_documents.extend(documents)
        
        for file in new_files:
            with open(file.name, "wb") as f:
                f.write(file.read())
            
            if file_type == "pdf":
                documents = load_pdf(file.name)
            elif file_type == "excel":
                documents = load_excel(file)
            else:
                st.error(f"Unsupported file type: {file_type}")
                return "Unsupported file type"
            
            if not documents:
                st.error(f"Failed to load file: {file.name}")
                return f"Failed to load file: {file.name}"
            all_documents.extend(documents)

        st.info(f"Splitting {len(all_documents)} documents...")
        split_docs = split_documents(all_documents)
        
        if self.vector_store is None:
            self.vector_store = create_vector_store(split_docs)
        else:
            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
            self.vector_store.add_documents(split_docs)
        
        if not self.vector_store:
            st.error("Failed to create vector store")
            return "Failed to create vector store"
        
        st.info("Creating QA chain...")
        self.qa_chain = create_qa_chain(self.vector_store)
        if not self.qa_chain:
            st.error("Failed to create QA chain")
            return "Failed to create QA chain"
        
        self.processed_files.extend([f.name for f in new_files])
        
        st.success(f"Successfully processed {len(new_files)} file(s).")
        return f"Successfully processed {len(new_files)} file(s). Total processed files: {len(self.processed_files)}"
    
    # Query the processed documents
    def query_documents(self, query: str):
        if not self.qa_chain:
            return "Please upload and process files first", []
        
        try:
            response = self.qa_chain.invoke({"query": query})
            if not response:
                return "No response from the QA chain.", []
            
            return response['result'], []
        
        except Exception as e:
            st.error(f"Error processing query: {e}")
            return f"Error processing query: {e}", []

def main():
    st.title("Eshwaran's AI Assistant")
    st.markdown("### Upload Documents (PDF, URL, Excel) and ask questions about their content.")

    if "document_processor" not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()
    document_processor = st.session_state.document_processor
    
    uploaded_files = st.file_uploader("Upload Files (PDF, Excel)", type=["pdf", "csv", "xlsx"], accept_multiple_files=True)
    file_type = st.radio("Select File Type", ["pdf", "excel"], index=0)

    url = st.text_input("Enter URL for Processing", "")
    
    if st.button("Process Documents"):
        if uploaded_files or url:
            status = document_processor.process_documents(uploaded_files, file_type, url)
            st.success(status)
        else:
            st.warning("Please upload files or enter a URL.")
    
    query = st.text_input("Ask a Question", "")
    if st.button("Submit Query"):
        if query.strip():
            answer, sources = document_processor.query_documents(query)
            if answer:
                st.text_area("Answer", value=answer, height=100, disabled=True)
            else:
                st.warning("No answer returned. Ensure files are processed correctly.")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
