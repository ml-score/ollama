# Ollama / Llama3
# Work with (local) Ollama and Llama large language models
# https://github.com/ml-score/ollama

# in order to be used locally you should download a transformer model to create vector stores from your files

from transformers import AutoModel, AutoTokenizer
import os
import requests

# Proxy configuration
proxy = "http://proxy.my-company.com:8080"  # Replace with your proxy server and port
proxy = ""
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy

def download_and_save_model(model_name, local_directory):
    # Create the directory if it does not exist
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)
    
    # Download and save the model and tokenizer locally
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(local_directory)
    tokenizer.save_pretrained(local_directory)
    print(f"Model and tokenizer saved to {local_directory}")

if __name__ == "__main__":
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    local_directory = "../data/local_models/all-MiniLM-L6-v2"
    download_and_save_model(model_name, local_directory)
