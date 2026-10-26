from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import faiss
import numpy as np
import uvicorn
import os
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor

crawl_lock = threading.Lock()

# =====================LOAD ENV===========================

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")


# =====================CONFIG===========================

BASE_URL = "https://www.honda-mideast.com/en/"

MAX_PAGES = 200
CRAWL_DEPTH = 3
TOP_K_RESULTS = 8


# ====================FASTAPI==========================

app = FastAPI(
    title="Honda Middle East AI Knowledge Base",
    version="2.0"
)

# ==================GLOBAL STORAGE=========================

documents = []
metadata_store = []
visited = set()


conversation_memory = {}

# ============EMBEDDING MODEL==========================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ===============VECTOR DB=========================

sample_embedding = embedding_model.encode(["test"])

dimension = len(sample_embedding[0])

index = faiss.IndexFlatL2(dimension)

# ================TEXT SPLITTER=======================

class SimpleTextSplitter:
    def __init__(self, chunk_size=800, chunk_overlap=150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        if not text:
            return []
        chunks = []
        start = 0
        text_len = len(text)
        step = max(1, self.chunk_size - self.chunk_overlap)
        while start < text_len:
            end = min(text_len, start + self.chunk_size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_len:
                break
            start += step
        return chunks

text_splitter = SimpleTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

# ============REQUEST MODEL=========================

class ChatRequest(BaseModel):
    session_id: str
    question: str

# ==============CLEAN TEXT============================

def clean_text(text):

    # Strip repeated navigation bar and boilerplate text
    text = re.sub(
        r'OUR PRODUCTS\s+AUTOMOBILES\s+MOTORCYCLES.*?SAFETY & TECHNOLOGY\s+HONDA SUSTAINABILITY',
        ' ',
        text,
        flags=re.IGNORECASE
    )
    text = text.replace("Loading component...", " ")
    text = re.sub(r'\s+', ' ', text)
    text = text.replace("\n", " ")
    text = text.strip()

    return text

# ==========URL VALIDATION=============================

def is_valid_url(url):

    parsed = urlparse(url)

    if parsed.netloc != "www.honda-mideast.com":
        return False

    url = url.lower().rstrip("/")

    blocked_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".svg",
        ".pdf",
        ".zip",
        ".mp4",
        ".webp",
        ".ashx"
    )

    if url.endswith(blocked_extensions):
        return False

    blocked_keywords = [
        "automobiles",
        "motorcycle",
        "marine",
        "power-products",
        "comparison",
        "specification",
        "media-center",
        "facebook",
        "instagram",
        "youtube",
        "linkedin",
        "twitter",
        "privacy-policy"
    ]

    if any(keyword in url for keyword in blocked_keywords):
        return False

    allowed_keywords = [
        "/en/",
        "about-us",
        "history",
        "news",
        "discover",
        "safety",
        "technology",
        "honda-sensing",
        "sustainability"
    ]

    return any(keyword in url for keyword in allowed_keywords)


# ===============SCRAPER==========================

def scrape_page(url):

    try:

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=30
        )

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "noscript"
        ]):
            tag.decompose()

        text = soup.get_text(separator=" ")

        text = clean_text(text)

        links = []

        for a in soup.find_all("a", href=True):

            full_url = urljoin(url, a["href"])

            full_url = full_url.split("#")[0]

            if is_valid_url(full_url):
                links.append(full_url)

        return {
            "url": url,
            "text": text,
            "links": list(set(links))
        }

    except Exception as e:

        print(f"Error scraping {url}: {e}")

        return None

# ================= CRAWLER ========================

def crawl(url, depth=0):

    url = url.rstrip("/")

    if depth > CRAWL_DEPTH:
        return

    with crawl_lock:
        if url in visited:
            return
        if len(visited) >= MAX_PAGES:
            return
        visited.add(url)

    print(f"Crawling: {url}")

    data = scrape_page(url)

    if not data or not data.get("text"):
        return

    chunks = text_splitter.split_text(data["text"])

    with crawl_lock:
        for chunk in chunks:
            documents.append(chunk)
            metadata_store.append({
                "source": data["url"]
            })

    if depth < CRAWL_DEPTH:
        for link in data.get("links", []):
            clean_link = link.rstrip("/")
            with crawl_lock:
                already_visited = clean_link in visited
            if not already_visited:
                crawl(clean_link, depth + 1)

# ===============BUILD VECTOR DB======================

def build_vector_db():

    global index

    if not documents:

        print("No documents found to index")

        return

    print("\nGenerating embeddings...\n")

    embeddings = embedding_model.encode(
        documents,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings).astype("float32")

    index.add(embeddings)

    print("\nSaving FAISS index...\n")

    faiss.write_index(
        index,
        "honda_index.faiss"
    )

    with open(
        "documents.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "documents": documents,
                "metadata": metadata_store
            },
            f,
            ensure_ascii=False
        )

    print("\nVector DB Ready\n")

# ============== RETRIEVAL ======================

def retrieve_context(question):

    if index.ntotal == 0:
        return []

    question_embedding = embedding_model.encode([question])

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    distances, indices = index.search(
        question_embedding,
        TOP_K_RESULTS
    )

    retrieved_chunks = []

    for idx in indices[0]:

        if 0 <= idx < len(documents):

            retrieved_chunks.append({
                "content": documents[idx],
                "metadata": metadata_store[idx]
            })

    return retrieved_chunks

#========== Conversation Memory ==================

def get_conversation_memory(session_id):

    if session_id not in conversation_memory:

        conversation_memory[session_id] = []

    return conversation_memory[session_id]


# =========== DEEPSEEK LLM =====================

def ask_deepseek(session_id, question, context):

    combined_context = "\n\n".join([
        item["content"] for item in context
    ])
    
    memory = get_conversation_memory(session_id)

    conversation_history = ""

    for item in memory:

        conversation_history += f"""
    User: {item['user']}
    Assistant: {item['assistant']}
    """

    prompt = f"""
You are the official Honda AI Knowledge Assistant.

CRITICAL RULE - STRICT WEBSITE CONTENT GROUNDING (MANDATORY):
1. You must answer ONLY and EXCLUSIVELY using the facts, specifications, and details provided in the "Website Knowledge Context" below (extracted directly from the official Honda website content pages).
2. DO NOT use external general knowledge, assumptions, or information not present in the Website Knowledge Context.
3. If the answer to the user's question cannot be found or verified in the Website Knowledge Context below, you MUST NOT guess or make up an answer. State clearly:
   "I could not find information on this in the official Honda website pages. Please check the official Honda website (www.honda-mideast.com) or contact your nearest authorized Honda dealer."
4. If the question is outside the scope of Honda (e.g. general science, electricity transformers, AI/machine learning, biology, other car brands, trivia, etc.), refuse immediately with:
   "I am specifically designed to assist with Honda products, vehicles, technologies, and services based on official website documentation. I cannot assist with topics outside of Honda."

Your behavior:
- Answer like a knowledgeable assistant using ONLY the facts explicitly stated in the website content
- Keep responses concise, structured, and summarized (use short bullet points)
- Do NOT output long paragraphs, disclaimers, or walls of text
- Ground every single specification, model, and feature directly in the Website Knowledge Context below

Website Knowledge Context (from official Honda website pages):
{combined_context}

Conversation History:
{conversation_history}

User Question:
{question}

Answer based ONLY on the Website Knowledge Context above:
"""

    url = "https://api.deepseek.com/chat/completions"

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "You are the official Honda AI Knowledge Assistant. You answer questions strictly and exclusively from the provided official website content pages. If the information is not present in the website context, state that it is not available in the website content."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.0,
        "max_tokens": 450
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload
    )

    if response.status_code != 200:

        raise HTTPException(
            status_code=500,
            detail=response.text
        )

    result = response.json()

    answer = result["choices"][0]["message"]["content"]

    memory.append({
        "user": question,
        "assistant": answer
    })

    if len(memory) > 10:
        memory.pop(0)

    return answer

# ================= STARTUP ====================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "honda_index.faiss")
DOCS_FILE = os.path.join(BASE_DIR, "documents.json")

@app.on_event("startup")
def startup_event():

    global documents
    global metadata_store
    global index

    if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):

        print("\nLoading existing FAISS index and documents...\n")

        index = faiss.read_index(INDEX_FILE)

        with open(DOCS_FILE, "r", encoding="utf-8") as f:

            data = json.load(f)

            documents = data.get("documents", [])

            metadata_store = data.get("metadata", [])

        print(f"Knowledge Base Loaded ({len(documents)} document chunks)\n")

    else:

        print("\nNo existing index found, building knowledge base...\n")

        crawl(BASE_URL)

        build_vector_db()

    print("Knowledge Base Ready\n")

# Auto-initialize on import so memory is always loaded
startup_event()


# =========== ROOT ===================

@app.get("/")
def root():

    return {
        "status": "running",
        "documents_indexed": len(documents),
        "pages_crawled": len(visited),
        "vector_db_size": index.ntotal
    }

# ============== CHAT API ======================

@app.post("/knowledge_chat")
def chat(request: ChatRequest):

    session_id = request.session_id

    question = request.question

    context = retrieve_context(question)

    answer = ask_deepseek(
        session_id,
        question,
        context or []
    )

    sources = []
    if context:
        for item in context:
            src = item.get("metadata", {}).get("source", "")
            if src and src not in sources:
                sources.append(src)

    return {
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "sources": sources
    }

