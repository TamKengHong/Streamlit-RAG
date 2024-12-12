# Streamlit-RAG: Multi-document RAG Q&A with Langchain + LanceDB + LMStudio. 

Works completely offline and you can use your own local models. We use LM Studio llama3.2 for the LLM and nomic-embed-text for the embedder.

Runs on Python 3.12

# Setup instructions:
1. git clone this repository
2. pip install -r requirements.txt
3. Start your LM Studio server and modify the server.py code to point to the ip address for the LLM and Embedder. You may choose to use other models if needed.
4. Run the command on terminal: streamlit run server.py

# Features Demo:
