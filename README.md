# Multi PDF, excel, URL Question Answering System using LLAMA 3.3 and GROQ Cloud
## Overview
#### This is a Python-based PDF Question Answering (QA) system that allows users to upload multiple PDF files and ask questions about their contents. The application uses advanced natural language processing techniques to extract and retrieve relevant information from the uploaded documents.
#Features
  Multiple PDF, Excel or URL file upload
  Document text extraction and chunking
  Vector-based semantic search
  Question answering using state-of-the-art language model
  Gradio-based web interface for easy interaction

## Technologies Used
Gradio: Web interface creation
LangChain: Document processing and QA workflow
HuggingFace: Embedding generation
Groq: Large Language Model (LLM) for question answering
Fiass: Vector store for document embeddings

## How It Works
PDF documents are loaded and split into smaller chunks
Document chunks are converted to vector embeddings
When a query is submitted, semantic similarity search finds relevant document chunks
The language model generates an answer based on the retrieved chunks
## Configuration
Adjust chunk_size and chunk_overlap in split_documents() to fine-tune document processing
Change the embedding model in create_vector_store()
Modify Groq LLM parameters in create_qa_chain()
## Dependencies
gradio
python-dotenv
langchain
transformers
chromadb
groq
## Limitations
Performance depends on the quality and complexity of input PDFs
Large documents may require more computational resources
Accuracy varies based on the chosen embedding and language models
Contributing
Contributions are welcome! Please submit pull requests or open issues on the GitHub repository.

## Acknowledgments
LangChain
HuggingFace
Groq
Gradio
