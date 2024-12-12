import streamlit as st
import os
import pickle
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import LanceDB
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import lancedb
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
import requests
from typing import Optional, List, Mapping, Any


# Set up the Streamlit app
st.title("Multi-document RAG with Langchain + LanceDB + LMStudio")

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Directory to store uploaded documents
upload_dir = "./uploaded_files"
os.makedirs(upload_dir, exist_ok=True)

# Path to save chat history
chat_history_path = os.path.join(upload_dir, 'chat_history.pkl')

# Load chat history if it exists
if os.path.exists(chat_history_path):
    with open(chat_history_path, 'rb') as f:
        st.session_state['chat_history'] = pickle.load(f)

# File uploader for PDF, DOCX, and TXT documents
uploaded_files = st.file_uploader(
    "Upload your documents (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# Initialize variables
documents = []
vectorstore_dir = "./lancedb_vectorstore"  # Directory for LanceDB persistence

# Load existing documents from the upload directory
existing_files = [
    os.path.join(upload_dir, f)
    for f in os.listdir(upload_dir)
    if f.endswith(('.pdf', '.docx', '.txt'))
]

# Initialise embeddings
class LMStudioEmbeddings(Embeddings):
    def __init__(self, api_url: str, model_name: str):
        self.api_url = api_url
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        headers = {'Content-Type': 'application/json'}
        payload = {
            'model': self.model_name,
            'input': texts
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"Embedding request failed with status {response.status_code}: {response.text}")

        data = response.json()
        # Extract the embeddings
        # Ensure `data['data']` contains items with 'embedding' keys
        embeds = [item['embedding'] for item in data['data']]
        return embeds

    def embed_query(self, text: str) -> List[float]:
        # This will return a single embedding vector for the query
        return self.embed_documents([text])[0]


embeddings = LMStudioEmbeddings(
    api_url='http://localhost:1234/v1/embeddings',   # Adjust to your LMStudio embedding endpoint
    model_name='text-embedding-nomic-embed-text-v1.5'                 # Adjust to your embedding model name
)

# Check if the vectorstore directory exists
if os.path.exists(vectorstore_dir):
    # Connect to the existing LanceDB database
    db = lancedb.connect(vectorstore_dir)
    # Load existing LanceDB vectorstore
    vectorstore = LanceDB(
        connection=db,
        table_name="vector_table",
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever()
    st.success("Loaded existing vector store from disk.")
else:
    vectorstore = None
    # Create the LanceDB directory if it doesn't exist
    os.makedirs(vectorstore_dir, exist_ok=True)
    # Connect to a new LanceDB database
    db = lancedb.connect(vectorstore_dir)

# Process existing documents if vectorstore doesn't exist
if not vectorstore:
    st.info("Processing existing documents...")
    for file_path in existing_files:
        file_name = os.path.basename(file_path)
        # Determine file type and use appropriate loader
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            st.warning(f"Unsupported file type: {file_name}")
            continue

        # Load data and add document name to metadata
        data = loader.load()
        for doc in data:
            doc.metadata["source"] = file_name
        documents.extend(data)

# Process newly uploaded files
if uploaded_files:
    st.header("Uploaded Documents")
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_path = os.path.join(upload_dir, file_name)
        # Save uploaded files to the local directory
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded: {file_name}")

        # Determine file type and use appropriate loader
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(file_path)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            loader = Docx2txtLoader(file_path)
        elif uploaded_file.type == "text/plain":
            loader = TextLoader(file_path)
        else:
            st.warning(f"Unsupported file type: {uploaded_file.type}")
            continue

        # Load data and add document name to metadata
        data = loader.load()
        for doc in data:
            doc.metadata["source"] = file_name
        documents.extend(data)

    # Set vectorstore to None to indicate that it needs to be rebuilt
    vectorstore = None

# If there are new or existing documents to process
if documents:
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_splits = text_splitter.split_documents(documents)

    # Initialize LanceDB vectorstore
    vectorstore = LanceDB.from_documents(
        documents=all_splits,
        embedding=embeddings,
        connection=db,
        table_name="vector_table"
    )
    st.success("Created new vector store.")

    retriever = vectorstore.as_retriever()
elif vectorstore:
    # Use existing retriever
    retriever = vectorstore.as_retriever()
else:
    st.info("No documents available for querying. Please upload documents.")
    st.stop()

# Provide option to list and download uploaded files
uploaded_docs = [
    f for f in os.listdir(upload_dir)
    if f.endswith(('.pdf', '.docx', '.txt'))
]
if uploaded_docs:
    st.write("### Available Documents:")
    for doc in uploaded_docs:
        with open(os.path.join(upload_dir, doc), "rb") as f:
            st.download_button(
                label=f"Download {doc}",
                data=f.read(),
                file_name=doc
            )

# Prompt template
template = """You are an AI assistant that provides concise and direct answers based solely on the provided context.

Context:
{context}

Question: {question}

Answer without listing options or choices."""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)


class LLMStudioLLM(LLM):
    api_key: str = ''  # Add your API key if required
    model_name: str = 'llama-3.2-3b-instruct'  # Use your actual model name
    api_url: str = 'http://localhost:1234/v1/completions'  # Use the LMStudio server IP

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {'Content-Type': 'application/json'}
        payload = {
            'prompt': prompt,
            'model': self.model_name,
            'max_tokens': 512,
            'temperature': 0.5,
            'top_p': 1.0,
            'n': 1,
            'stream': False,
            'logprobs': None,
            'stop': stop,
        }
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        response = requests.post(self.api_url, headers=headers, json=payload)
        print(f"Response Status Code: {response.status_code}")
        print(f"Response Text: {response.text}")
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['text'].strip()
        else:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "LLMStudioLLM"

# Initialize LLMStudio LLM
model_local = LLMStudioLLM(
    api_key='',  # Add your API key if required
    model_name='llama-3.2-3b-instruct',
    api_url='http://localhost:1234/v1/completions'  # Use the LMStudio server IP
)

# Create the LLM chain
llm_chain = LLMChain(llm=model_local, prompt=prompt)

# Query input
query = st.text_input("Enter your query:")
if query:
    # Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(query)

    if retrieved_docs:
        # Combine contexts
        contexts = [doc.page_content for doc in retrieved_docs]
        sources = [doc.metadata["source"] for doc in retrieved_docs]
        combined_context = "\n".join(contexts)

        # Generate answer
        with st.spinner("Generating response..."):
            response = llm_chain.run(context=combined_context, question=query)
        st.write("### Answer:")
        st.write(response)

        # Append to chat history
        st.session_state.chat_history.append({'query': query, 'response': response})

        # Save chat history to disk
        with open(chat_history_path, 'wb') as f:
            pickle.dump(st.session_state['chat_history'], f)

        # Display chat history with chat bubbles
        st.write("### Chat History:")
        for chat in st.session_state.chat_history:
            # Display user's message on the right
            with st.chat_message("user"):
                st.markdown(chat['query'])
            # Display AI's message on the left
            with st.chat_message("assistant"):
                st.markdown(chat['response'])

        # Display contexts in a dropdown
        st.write("### Retrieved Contexts:")
        options = [f"Context {i+1} from {sources[i]}" for i in range(len(contexts))]
        selected_option = st.selectbox("Select a context to view:", options)
        selected_index = options.index(selected_option)
        selected_context = contexts[selected_index]
        st.write("#### Selected Context:")
        st.write(selected_context)

        # Provide a download button for the source document
        file_path = os.path.join(upload_dir, sources[selected_index])
        with open(file_path, "rb") as f:
            st.download_button(
                label=f"Download {sources[selected_index]}",
                data=f.read(),
                file_name=sources[selected_index],
                key=f"{sources[selected_index]}"
            )
    else:
        st.write("No relevant documents found for the query.")

    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        if os.path.exists(chat_history_path):
            os.remove(chat_history_path)
        st.success("Chat history cleared.")
else:
    st.info("Please enter a query to get started.")
