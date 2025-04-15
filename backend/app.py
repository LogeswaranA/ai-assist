from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import os
from dotenv import load_dotenv
from collections import defaultdict
import uuid
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="AI Plugin Assist API")

# Load environment variables
load_dotenv(override=True)

# Define request body model
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

# Define response model
class QueryResponse(BaseModel):
    response: str
    session_id: str

# In-memory session storage
session_history = defaultdict(list)

# File paths for stored embeddings and chunks
INDEX_FILE = "context_index.faiss"
CHUNKS_FILE = "context_chunks.pkl"
TEXT_FILE = "llms.txt"

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to load or create embeddings and FAISS index
def load_or_create_embeddings():
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        logger.info("Loading precomputed embeddings and index...")
        # Load FAISS index
        index = faiss.read_index(INDEX_FILE)
        # Load chunks
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
        logger.info("Loaded embeddings and chunks from disk.")
    else:
        logger.info("Creating new embeddings and index...")
        # Read and process llms.txt
        try:
            with open(TEXT_FILE, "r") as file:
                raw_context = file.read()
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail=f"{TEXT_FILE} file not found")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(raw_context)

        # Generate embeddings
        chunk_embeddings = embedding_model.encode(chunks, convert_to_numpy=True)

        # Create FAISS index
        dimension = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(chunk_embeddings)

        # Save index and chunks to disk
        faiss.write_index(index, INDEX_FILE)
        with open(CHUNKS_FILE, "wb") as f:
            pickle.dump(chunks, f)
        logger.info(f"Saved embeddings to {INDEX_FILE} and chunks to {CHUNKS_FILE}.")

    return index, chunks

# Load embeddings and chunks at startup
index, chunks = load_or_create_embeddings()

# Set up OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise HTTPException(status_code=500, detail="OPENAI_API_KEY environment variable not set")

# Initialize OpenAI model
llm = OpenAI(api_key=openai_api_key)

# Define prompt template
template = """
You are a helpful assistant for Plugin EcoSystem. Use the following context and conversation history to answer the user's query. If the query is unrelated, respond appropriately.

Context: {context}

Conversation History:
{history}

User Query: {query}

Answer:
"""
prompt = PromptTemplate(input_variables=["context", "history", "query"], template=template)

# Define the state for LangGraph
class ChatbotState(TypedDict):
    query: str
    context: str
    history: str
    response: str
    session_id: str

# Define the node function for processing the query
def process_query(state: ChatbotState) -> ChatbotState:
    try:
        # Embed the query
        query_embedding = embedding_model.encode([state["query"]], convert_to_numpy=True)
        
        # Search for top-k relevant chunks
        k = 3
        distances, indices = index.search(query_embedding, k)
        relevant_chunks = [chunks[idx] for idx in indices[0]]
        context = "\n".join(relevant_chunks)

        # Format the prompt
        formatted_prompt = prompt.format(
            context=context,
            history=state["history"],
            query=state["query"]
        )
        
        # Call LLM
        response = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": formatted_prompt}],
            max_tokens=1000,
            temperature=0.5
        )
        return {
            "response": response.choices[0].message.content.strip(),
            "query": state["query"],
            "context": context,
            "history": state["history"],
            "session_id": state["session_id"]
        }
    except Exception as e:
        logger.error(f"Error in process_query: {e}")
        return {
            "response": "Sorry, an error occurred while processing your query.",
            "query": state.get("query", ""),
            "context": state.get("context", ""),
            "history": state.get("history", ""),
            "session_id": state.get("session_id", "")
        }

# Build the LangGraph workflow
workflow = StateGraph(ChatbotState)
workflow.add_node("process_query", process_query)
workflow.set_entry_point("process_query")
workflow.add_edge("process_query", END)
chatbot_graph = workflow.compile()

# API endpoint to handle queries
@app.post("/query", response_model=QueryResponse)
async def get_response(request: QueryRequest):
    try:
        # Assign or use session_id
        session_id = request.session_id or str(uuid.uuid4())

        # Get conversation history for this session
        history = session_history[session_id]
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

        # Run the LangGraph workflow
        initial_state = {
            "query": request.query,
            "context": "",
            "history": history_str,
            "response": "",
            "session_id": session_id
        }
        result = chatbot_graph.invoke(initial_state)

        # Update session history
        session_history[session_id].append({"role": "user", "content": request.query})
        session_history[session_id].append({"role": "assistant", "content": result["response"]})

        # Limit history size
        if len(session_history[session_id]) > 20:
            session_history[session_id] = session_history[session_id][-20:]

        return {"response": result["response"], "session_id": session_id}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "AI Plugin Assist API is running"}

# Optional: Endpoint to force reindexing if llms.txt changes
@app.post("/reindex")
async def reindex():
    try:
        if os.path.exists(INDEX_FILE):
            os.remove(INDEX_FILE)
        if os.path.exists(CHUNKS_FILE):
            os.remove(CHUNKS_FILE)
        global index, chunks
        index, chunks = load_or_create_embeddings()
        return {"message": "Reindexing complete"}
    except Exception as e:
        logger.error(f"Error during reindexing: {e}")
        raise HTTPException(status_code=500, detail=f"Error reindexing: {str(e)}")