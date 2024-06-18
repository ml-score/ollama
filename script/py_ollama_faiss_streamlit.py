# Ollama / Llama3
# Work with (local) Ollama and Llama large language models
# https://github.com/ml-score/ollama

import streamlit as st  # Streamlit is used to create the web app interface
from PyPDF2 import PdfReader  # PyPDF2 is used to read PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits text into manageable chunks

# pip install -U langchain-community
from langchain_community.vectorstores import FAISS  # Vector store to store and retrieve text embeddings using FAISS
from langchain_community.document_loaders import TextLoader, CSVLoader  # Loaders for text and CSV documents
from langchain_community.embeddings import OllamaEmbeddings  # Creates embeddings for text using different models
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama  # Large Language Model interface for Ollama

from langchain.prompts import ChatPromptTemplate  # Creates templates for prompts to the LLM
from langchain.chains import RetrievalQA  # Creates a QA chain using retrieval-based methods
from langchain.schema import Document  # Document schema for text data
import os  # OS module for environment configuration and file handling
import tempfile  # Create temporary files
import json  # JSON module for reading and writing JSON data
from datetime import datetime  # Handles date and time operations

# Directory for local models
local_model_dir = "../data/local_models/all-MiniLM-L6-v2"

# Function to check if the model is downloaded
def check_model_downloaded():
    if not os.path.exists(local_model_dir):
        raise FileNotFoundError(f"The model directory '{local_model_dir}' does not exist. Please run the download_model.py script to download the model.")

# Call the function to ensure the model is downloaded
check_model_downloaded()

# Configure proxy settings if needed
def configure_proxy(use_proxy):
    proxy = "http://proxy.my-company.com:8080" if use_proxy else ""
    os.environ['http_proxy'] = proxy
    os.environ['https_proxy'] = proxy

# Read data from uploaded files
def read_data(files, loader_type):
    documents = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name

        try:
            if loader_type == "PDF":
                pdf_reader = PdfReader(tmp_file_path)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    documents.append(Document(page_content=text))
            elif loader_type == "Text":
                loader = TextLoader(tmp_file_path)
                documents.extend(loader.load())
            elif loader_type == "CSV":
                loader = CSVLoader(tmp_file_path)
                documents.extend(loader.load())
        finally:
            os.remove(tmp_file_path)
    return documents

# Split text into chunks
def get_chunks(texts, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for text in texts:
        chunks.extend(text_splitter.split_text(text.page_content))
    return chunks

# Store text chunks in a vector store using FAISS
def vector_store(text_chunks, embedding_model_name, vector_store_path):
    if embedding_model_name == "all-MiniLM-L6-v2":
        model_kwargs = {'trust_remote_code': True}
        embeddings = HuggingFaceEmbeddings(model_name=local_model_dir,  model_kwargs=model_kwargs)
    else:
        embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model=embedding_model_name)

    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local(vector_store_path)

# Load the vector store using FAISS
def load_vector_store(embedding_model_name, vector_store_path):
    if embedding_model_name == "all-MiniLM-L6-v2":
        model_kwargs = {'trust_remote_code': True}
        embeddings = HuggingFaceEmbeddings(model_name=local_model_dir,  model_kwargs=model_kwargs)
    else:
        embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model=embedding_model_name)
    
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store

# Save conversation history to a JSON file
def save_conversation(conversation, vector_store_path):
    conversation_path = os.path.join(vector_store_path, "conversation_history.json")
    with open(conversation_path, "w") as f:
        json.dump(conversation, f, indent=4)

# Load conversation history from a JSON file
def load_conversation(vector_store_path):
    conversation_path = os.path.join(vector_store_path, "conversation_history.json")
    if os.path.exists(conversation_path):
        with open(conversation_path, "r") as f:
            conversation = json.load(f)
    else:
        conversation = []
    return conversation

# Get a conversational chain response from the LLM
def get_conversational_chain(retriever, ques, llm_model, system_prompt):
    llm = Ollama(model=llm_model, verbose=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    response = qa_chain.invoke({"query": ques})
    return response

# Handle user input and display the response
def user_input(user_question, embedding_model_name, vector_store_path, num_docs, llm_model, system_prompt):
    vector_store = load_vector_store(embedding_model_name, vector_store_path)
    retriever = vector_store.as_retriever(search_kwargs={"k": num_docs})
    response = get_conversational_chain(retriever, user_question, llm_model, system_prompt)
    
    conversation = load_conversation(vector_store_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if 'output' in response:
        result = response['output']['result'] if 'result' in response['output'] else response['output']
        conversation.append({"question": user_question, "answer": result, "timestamp": timestamp, "llm_model": llm_model})
        st.write("Reply: ", result)
    else:
        conversation.append({"question": user_question, "answer": response, "timestamp": timestamp, "llm_model": llm_model})
        st.write("Reply: ", response)
    
    save_conversation(conversation, vector_store_path)
    
    st.write("### Conversation History")
    for entry in sorted(conversation, key=lambda x: x['timestamp'], reverse=True):
        st.write(f"**Q ({entry['timestamp']}):** {entry['question']}")
        st.write(f"**A:** {entry['answer']}")
        st.write(f"**LLM Model:** {entry['llm_model']}")

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Ollama: Chat with your Files")
    st.header("Chat with Your Files using Llama3 or Mistral")

    st.sidebar.markdown(
        """
        <div style="text-align: center;">
            <img src="https://ollama.com/public/ollama.png" alt="Ollama Logo" style="width: 50px; height: auto;">
            <p><b>Ollama with FAISS Vector Store</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Add system prompt input field
    system_prompt = st.sidebar.text_area("System Prompt", value="You are a helpful assistant. Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in the provided context, just say, 'answer is not available in the context', don't provide the wrong answer")

    user_question = st.text_input("Ask a Question from the Files")
    use_proxy = st.sidebar.checkbox("Use Proxy", value=False)
    configure_proxy(use_proxy)

    embedding_model_name = st.sidebar.selectbox(
        "Select Embedding Model",
        ["all-MiniLM-L6-v2", "llama3:instruct", "mistral:instruct"]
    )

    llm_model = st.sidebar.selectbox(
        "Select LLM Model",
        ["llama3:instruct", "mistral:instruct"]
    )

    vector_store_path = st.sidebar.text_input("Vector Store Path (will reload if there)", "../data/vectorstore/my_store")
    data_type = st.sidebar.selectbox(
        "Select Data Type",
        ["PDF", "Text", "CSV"]
    )

    chunk_text = st.sidebar.checkbox("Chunk Text", value=True)
    chunk_size = st.sidebar.number_input("Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
    chunk_overlap = st.sidebar.number_input("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
    num_docs = st.sidebar.number_input("Number of Documents to Retrieve", min_value=1, max_value=10, value=3, step=1)

    if user_question:
        user_input(user_question, embedding_model_name, vector_store_path, num_docs, llm_model, system_prompt)

    with st.sidebar:
        st.title("Documents:")
        data_files = st.file_uploader("Upload your Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_documents = read_data(data_files, data_type)
                if chunk_text:
                    text_chunks = get_chunks(raw_documents, chunk_size, chunk_overlap)
                else:
                    text_chunks = [doc.page_content for doc in raw_documents]
                vector_store(text_chunks, embedding_model_name, vector_store_path)
                st.success("Done")
    
    # Footer with three columns
    st.markdown("<hr>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.write("composed by [M. Lauber](https://medium.com/@mlxl) and ChatGPT-4o - [GitHub](https://github.com/ml-score/ollama)")  
    with col2:
        st.write("inspired by [Paras Madan](https://medium.com/@parasmadan.in)")

if __name__ == "__main__":
    main()
