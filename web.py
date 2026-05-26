from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import faiss
import numpy as np
import uvicorn
import os
import json
import re
from concurrent.futures import ThreadPoolExecutor

# =====================LOAD ENV===========================

load_dotenv()

DEEPSEEK_API_KEY = "sk-b8d54f473602495ca415df013c0892aa"


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

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

# ============REQUEST MODEL=========================

class ChatRequest(BaseModel):
    session_id: str
    question: str

# ==============CLEAN TEXT============================

def clean_text(text):

    text = re.sub(r'\s+', ' ', text)

    text = text.replace("\n", " ")

    text = text.strip()

    return text

# ==========URL VALIDATION=============================

def is_valid_url(url):

    parsed = urlparse(url)

    if parsed.netloc != "www.honda-mideast.com":
        return False

    url = url.lower()


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
        "twitter"
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

    if depth > CRAWL_DEPTH:
        return

    if url in visited:
        return

    if len(visited) >= MAX_PAGES:
        return

    visited.add(url)

    print(f"Crawling: {url}")

    data = scrape_page(url)

    if not data:
        return

    chunks = text_splitter.split_text(data["text"])

    for chunk in chunks:

        documents.append(chunk)

        metadata_store.append({
            "source": data["url"]
        })

    with ThreadPoolExecutor(max_workers=3) as executor:

        futures = []

        for link in data["links"]:

            futures.append(
                executor.submit(crawl, link, depth + 1)
            )

        for future in futures:
            future.result()

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

        if idx < len(documents):

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
You are Honda Middle East's intelligent AI assistant.

Your behavior:
- Answer like a knowledgeable human assistant
- Sound natural and conversational
- Explain concepts clearly and confidently
- Do NOT sound robotic or like documentation
- Maintain conversation continuity
- Understand follow-up questions
- Use previous conversation memory
- Use ONLY the provided context
- Never invent information
- If information is missing, clearly say so
- Never hallucinate

Your goal:
Help users understand Honda technologies, safety systems, and innovations in a friendly and professional way.

Conversation History:
{conversation_history}

Knowledge Context:
{combined_context}

Current User Question:
{question}

Answer naturally:
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
                "content": "You are a highly intelligent Honda expert assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 1000
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

@app.on_event("startup")
def startup_event():

    global documents
    global metadata_store
    global index

    print("\nStarting Honda Knowledge Base Builder...\n")

    if os.path.exists("honda_index.faiss"):

        print("Loading existing FAISS index...\n")

        index = faiss.read_index(
            "honda_index.faiss"
        )

        with open(
            "documents.json",
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

            documents = data["documents"]

            metadata_store = data["metadata"]

        print("\nKnowledge Base Loaded\n")

    else:

        print("\nNo existing index found\n")

        crawl(BASE_URL)

        build_vector_db()

    print("\nKnowledge Base Ready\n")


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

    if not context:

        return {
            "session_id": session_id,
            "question": question,
            "answer": "Knowledge base is empty.",
            "sources": []
        }

    answer = ask_deepseek(
        session_id,
        question,
        context
    )

    sources = []

    for item in context:

        sources.append(
            item["metadata"]["source"]
        )

    return {
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "sources": list(set(sources))
    }

