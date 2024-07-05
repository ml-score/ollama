#### Build a simple Streamlit application to create a local vector store from your PDF, CSV and (unstructured) Text files

#### ... using FAISS or Chroma vector store

You will have to download and install Ollama (https://ollama.com/) and make sure you have the models ready. You have to run once a command like:

```
ollama run mistral:instruct
ollama pull llama3:instruct
```


* [py_ollama.yml](https://github.com/ml-score/ollama/blob/main/script/py_ollama.yml) - the configuration file for your [Python environment](https://medium.com/low-code-for-advanced-data-science/knime-and-python-setting-up-and-managing-conda-environments-2ac217792539)

You should then pull an embedding model (https://ollama.com/blog/embedding-models):

```
ollama pull mxbai-embed-large
```
  
* [py_ollama_chroma_streamlit.py](https://github.com/ml-score/ollama/blob/main/script/py_ollama_chroma_streamlit.py) - the Streamlit application for the Ollama [Vector Store with Chroma](https://github.com/chroma-core/chroma)
* [py_ollama_faiss_streamlit.py](https://github.com/ml-score/ollama/blob/main/script/py_ollama_faiss_streamlit.py) - the Streamlit application for the Ollama Vector Store with [FAISS](https://ai.meta.com/tools/faiss/)

```
cd C:\\Users\\x123456\\knime-workspace\\LLM_Space\\script

conda activate py_ollama

streamlit run py_ollama_chroma_streamlit.py

streamlit run py_ollama_faiss_streamlit.py
```
