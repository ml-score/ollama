import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
import os

# Proxy configuration
def configure_proxy(use_proxy):
    proxy = "http://proxy.my-company.com:8080" if use_proxy else ""
    os.environ['http_proxy'] = proxy
    os.environ['https_proxy'] = proxy

def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def vector_store(text_chunks, embedding_model_name):
    if embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2":
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    else:
        model = "llama3:instruct"
        embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model=model)

    documents = [Document(page_content=chunk) for chunk in text_chunks]
    vector_store = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory="vectorstore")
    vector_store.persist()

def load_vector_store(embedding_model_name):
    if embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2":
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    else:
        model = "llama3:instruct"
        embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model=model)
    
    vector_store = Chroma(persist_directory="vectorstore", embedding_function=embeddings)
    return vector_store

def get_conversational_chain(retriever, ques):
    model = "llama3:instruct"
    llm = Ollama(model=model, verbose=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in the provided context, just say, "answer is not available in the context", don't provide the wrong answer"""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, prompt=prompt)
    response = qa_chain.invoke({"query": ques})
    return response['output']

def user_input(user_question, embedding_model_name):
    vector_store = load_vector_store(embedding_model_name)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    response = get_conversational_chain(retriever, user_question)
    st.write("Reply: ", response)

def main():
    st.set_page_config("Chat PDF")
    st.header("RAG based Chat with PDF")

    user_question = st.text_input("Ask a Question from the PDF Files")
    use_proxy = st.sidebar.checkbox("Use Proxy", value=False)
    configure_proxy(use_proxy)

    embedding_model_name = st.sidebar.selectbox(
        "Select Embedding Model",
        ["sentence-transformers/all-MiniLM-L6-v2", "llama3:instruct"]
    )

    if user_question:
        user_input(user_question, embedding_model_name)

    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = pdf_read(pdf_doc)
                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks, embedding_model_name)
                st.success("Done")

if __name__ == "__main__":
    main()
