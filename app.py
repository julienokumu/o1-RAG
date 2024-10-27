# Import libraries
import gradio as gr # For creating the web interface
from huggingface_hub import InferenceClient # For accessing the LLM
from langchain.document_loaders import PyPDFLoader # For loading PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter # For splitting text into chunks
from langchain.vectorstores import FAISS # For creating vector embeddings database
from langchain.embeddings import HuggingFaceEmbeddings # For creating embeddings
import tempfile # For handling temporary files
from dotenv import load_dotenv
import os # For file operations

# Load environment variabes from .env file
load_dotenv

# Initialize the Hugging Face client with your API key
api_key = os.getenv("API_KEY")
client = InferenceClient(api_key=api_key)

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings() # This will use a default embedding model

# Initialize a variable to store our vector database
vector_store = None 

# Processing of the PDF file
def process_pdf(pdf_file):
    """
    Function to process uploaded PDF file and create vector embeddings
    """
    global vector_store

    # Check if a file was uploaded
    if pdf_file is None:
        return "Please upload a PDF file"

    # Get the file path directly(Gradio provides the file path)
    pdf_path = pdf_file.name

    try:
        # Load the PDF directly using the file path
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split the document inot chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000, # Number of characters per chunk
            chunk_overlap = 200 # Number of characters to overlap
        )
        texts = text_splitter.split_documents(documents)

        # Create vector store from the text chunks
        vector_store = FAISS.from_documents(texts, embeddings)

        return "PDF SUCCESSFULLY PROCESSED! You can now ask questions about it."

    except Exception as e:
        return f"Error processing PDF: {str(e)}"

# Response process
def generate_response(message, history):
    """
    Function to generate responses based on user input
    """
    global vector_store

    # Check if a PDF has been processed
    if vector_store is None:
        return "Please upload a PDF file first."

    # Check for relevant documents in the vector store
    docs = vector_store.similarity_search(message, k = 3) # Get top 3 most relevant chunks

    # Create context from retrieved documents
    context = "\n".join([doc.page_content for doc in docs])

    # Create the prompt for the LLM
    prompt = f"""Use the following context to answer the question. If the answer is not in the context, say "I don't have enough information to answer that."
    
    Context: {context}

    Question: {message}

    Answer:"""

    # Send prompt to LLM and get response
    messages = [{"role": "user", "content": prompt}]

    response = ""
    # Stream the response
    stream = client.chat.completions.create(
        model = "meta-llama/Llama-3.2-1B-Instruct",
        messages = messages,
        max_tokens = 500,
        stream = True
    )

    for chunk in stream:
        response += chunk.choices[0].delta.content or ""
        yield response

# Create the Gradio Web Interface
with gr.Blocks() as demo:
    # Add a title
    gr.Markdown("o1-RAG PDF Chat Assistant | Julien Okumu")

    # Add file upload component
    pdf_input = gr.File(
        label = "Upload your PDF",
        file_types = [".pdf"]
    )

    # Add upload button
    upload_button = gr.Button("Process PDF")

    # Add status text
    status_text = gr.Textbox(label = "Status")

    # Add Chat interface
    chatbot = gr.ChatInterface(
        generate_response,
        examples = ["What is this document about?", "Can you summarize the main points?"],
        title = "Chat with o1-RAG using your PDF"
    )

    # Handle PDF upload button click
    upload_button.click(
        fn = process_pdf,
        inputs = [pdf_input],
        outputs = [status_text]
    )

# Launch the interface
demo.launch(share=True)