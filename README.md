# Honda AI Unified Server 🏍️ 🚗

A unified FastAPI-powered AI platform featuring an **AI Bike Recommendation Consultant** and a **Honda Knowledge Base Assistant** with real-time word-by-word streaming.

---

## 📁 Repository Structure

| File | Description |
| :--- | :--- |
| **`main.py`** | Server entrypoint: mounts `/chat` and `/knowledge_chat`, provides `/status`, and serves the web UI at `/`. |
| **`app.py`** | **Recommendation Bot**: AI motorcycle consultant, bike aliases, and dynamic bike cards. |
| **`web.py`** | **Knowledge Bot**: Strictly answers Honda questions using website content at `temperature: 0.0`. |
| **`honda.html`** | **Frontend UI**: Responsive interface with typewriter streaming, animated cursor, and lead forms. |
| **`bikes_data.json`** | **Bike Cache**: Specs, images, and links for 21 Honda bikes (2-second startup without Chrome). |
| **`documents.json`** | **Knowledge Data**: Crawled text chunks from official Honda Middle East website pages. |
| **`honda_index.faiss`** | **Vector Database**: 384-dimensional FAISS index for neural similarity search. |
| **`requirements.txt`** | **Dependencies**: Clean list of Python packages needed to run the project. |
| **`.gitignore`** | **Security**: Keeps `.env`, Python cache, and virtual environments out of Git. |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the project root:
```ini
DEEPSEEK_API_KEY=your_deepseek_api_key_here
PORT=8000
```

### 3. Run the Server
```bash
python main.py
```
Visit **`http://localhost:8000`** in your browser.

---

## 🔌 API Endpoints

| Route | Method | Description |
| :--- | :--- | :--- |
| **`/`** | `GET` | Serves the interactive `honda.html` web application |
| **`/status`** | `GET` | Health check endpoint returning JSON server status |
| **`/chat/chat`** | `POST` | Bike Recommendation AI conversation endpoint |
| **`/knowledge_chat/knowledge_chat`** | `POST` | Honda Knowledge Base Q&A endpoint |
| **`/chat/new_session`** | `GET` | Generates a new unique session ID |
