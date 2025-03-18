import os
import pickle
import asyncio
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from playwright.async_api import async_playwright
import google.generativeai as genai
from dotenv import load_dotenv
import gc  # For garbage collection

# Load API Key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("Gemini API Key is missing. Please set it in the .env file.")

genai.configure(api_key=gemini_api_key)

# FastAPI App Initialization
app = FastAPI(title="Tripzoori AI Assistant API", description="API for answering questions about Tripzoori", version="1.0")

# Website URL
TRIPZOORI_URL = "https://tripzoori-gittest1.fly.dev/"
VECTOR_STORE_PATH = "vector_store.pkl"

# Function to Load and Process Website Content with Playwright
async def load_website_content_async(url):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Set timeout to avoid hanging
            await page.goto(url, wait_until="networkidle", timeout=30000)
            content = await page.content()
            await browser.close()
            
            document = Document(page_content=content, metadata={"source": url})
            
            # Memory efficient text splitting
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
            document_chunks = text_splitter.split_documents([document])
            
            # Use a smaller model for embeddings to reduce memory usage
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(document_chunks, embeddings)
            
            # Save vector store to disk
            with open(VECTOR_STORE_PATH, "wb") as f:
                pickle.dump(vector_store, f)
            
            # Force garbage collection
            del document, document_chunks
            gc.collect()
            
            return vector_store
    except Exception as e:
        raise Exception(f"Error fetching website content: {str(e)}")

# Function to Load Persisted Vector Store
def load_vector_store():
    if os.path.exists(VECTOR_STORE_PATH):
        try:
            with open(VECTOR_STORE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return None
    return None

# Prompt Template
prompt = PromptTemplate(
    input_variables=["context", "input"],
    template="""
    You are Tripzoori AI Assistant, designed to answer questions about the website 'tripzoori-gittest1.fly.dev'.
    Use the following context to provide accurate responses:

    {context}

    Question: {input}
    """
)

# Function to Generate Responses using Gemini
def generate_response(final_prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(final_prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Request Model for API Input
class QueryRequest(BaseModel):
    question: str

# GET API Endpoint with clear parameter requirements
@app.get("/ask", summary="Ask a question about Tripzoori")
async def ask_tripzoori_get(question: str = Query(..., description="Question about Tripzoori")):
    try:
        vector_store = load_vector_store()
        if not vector_store:
            vector_store = await load_website_content_async(TRIPZOORI_URL)
        
        # Reduce number of retrieved docs to save memory
        retrieved_docs = vector_store.similarity_search(question, k=2)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        final_prompt = prompt.format(context=context, input=question)
        response = generate_response(final_prompt)
        
        # Force garbage collection
        del retrieved_docs, context
        gc.collect()
        
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# POST API Endpoint
@app.post("/ask", summary="Ask a question about Tripzoori")
async def ask_tripzoori(request: QueryRequest):
    try:
        vector_store = load_vector_store()
        if not vector_store:
            vector_store = await load_website_content_async(TRIPZOORI_URL)
        
        # Reduce number of retrieved docs to save memory
        retrieved_docs = vector_store.similarity_search(request.question, k=2)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        final_prompt = prompt.format(context=context, input=request.question)
        response = generate_response(final_prompt)
        
        # Force garbage collection
        del retrieved_docs, context
        gc.collect()
        
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add a simple root endpoint to avoid 404 on /
@app.get("/")
async def root():
    return {"message": "Welcome to Tripzoori AI Assistant API. Use /ask endpoint to ask questions."}

# Add docs endpoints for better API documentation
@app.get("/docs")
async def get_docs():
    return {"message": "API documentation is available at /docs"}
