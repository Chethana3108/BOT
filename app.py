import os
import re as _re
import uuid
import threading
import requests
import uvicorn
import time
import json

from difflib import SequenceMatcher
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from bs4 import BeautifulSoup

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.documents import Document

# =====================================================
# SELENIUM IMPORTS
# =====================================================

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# =====================================================
# LOAD ENV
# =====================================================

load_dotenv()

DEEPSEEK_API_KEY = " "

# =====================================================
# FASTAPI APP
# =====================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# HONDA URLS
# =====================================================

BASE_URL = "https://www.honda-mideast.com"
LISTING_URL = "https://www.honda-mideast.com/en/motorcycle#listing"

# =====================================================
# SELENIUM DRIVER
# =====================================================

print("\nStarting Selenium Driver...")

options = Options()

options.add_argument("--headless")
options.add_argument("--window-size=1920,1080")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

# =====================================================
# SCRAPE HONDA WEBSITE
# =====================================================

print("\nFetching Honda bikes...")

html = requests.get(LISTING_URL).text

soup = BeautifulSoup(html, "lxml")

explore_links = set()

for a in soup.find_all("a", href=True):

    text = a.get_text(strip=True).upper()

    if "EXPLORE" in text:

        href = a["href"]

        if href.startswith("http"):
            explore_links.add(href)
        else:
            explore_links.add(BASE_URL + href)

print(f"\nFound {len(explore_links)} bike pages")

# =====================================================
# EXTRACT BIKE DETAILS + IMAGE
# =====================================================

documents = []

# Track seen titles to skip duplicate bike pages
seen_titles = set()

for url in explore_links:

    try:

        print("\nScraping:", url)

        # =========================================
        # NORMAL HTML SCRAPING
        # =========================================

        bike_html = requests.get(url).text

        bike_soup = BeautifulSoup(bike_html, "lxml")

        # =========================================
        # EXTRACT BIKE NAME FROM URL SLUG
        # =========================================

        slug = url.rstrip("/").split("/")[-1]

        slug_clean = _re.sub(r'^[Nn]ew[-\s]', '', slug)

        parts = _re.split(r'[-_\s]+', slug_clean)

        MODEL_CODES = {
            "cbr", "crf", "cbf", "cb", "gl", "nc", "trx", "cmx", "sp", "rr",
            "dct", "atv", "tm", "adv", "xadv"
        }

        def format_part(p):
            pl = p.lower()
            if pl in MODEL_CODES:
                return p.upper()
            if p.isalpha() and len(p) <= 3:
                return p.upper()
            if any(c.isdigit() for c in p):
                return p.upper()
            return p.capitalize()

        title = " ".join(format_part(p) for p in parts if p)

        print(f"  Title resolved: '{title}' (from slug: '{slug}')")

        if title in seen_titles:
            print(f"  Skipping duplicate: '{title}'")
            continue

        seen_titles.add(title)

        content_parts = []

        for tag in bike_soup.find_all(["h1", "h2", "h3", "p", "li"]):

            text = tag.get_text(" ", strip=True)

            if len(text) > 20:
                content_parts.append(text)

        content = "\n".join(content_parts)

        # =========================================
        # IMAGE EXTRACTION USING SELENIUM
        # =========================================

        image_url = ""

        try:

            design_url = url + "#design"

            driver.get(design_url)

            time.sleep(4)

            driver.execute_script("""
            document.getElementById('design').scrollIntoView();
            """)

            time.sleep(3)

            product_div = driver.find_element(
                By.CSS_SELECTOR,
                "#design .product .image"
            )

            img = product_div.find_element(By.TAG_NAME, "img")

            image_url = img.get_attribute("src")

            print("Image Found:", image_url)

        except Exception as img_error:

            print("Image extraction failed")
            print(img_error)

        # =========================================
        # CREATE DOCUMENT
        # =========================================

        doc = Document(
            page_content=content,
            metadata={
                "bike_name": title,
                "source": url,
                "image": image_url
            }
        )

        documents.append(doc)

    except Exception as e:

        print("Error scraping:", url)
        print(e)

# =====================================================
# CLOSE SELENIUM DRIVER
# =====================================================

driver.quit()

print(f"\nLoaded {len(documents)} bikes")

# =====================================================
# SAVE SCRAPED DATA
# =====================================================

bike_json_data = []

for d in documents:

    bike_json_data.append({
        "bike_name": d.metadata.get("bike_name"),
        "source": d.metadata.get("source"),
        "image": d.metadata.get("image"),
        "content": d.page_content
    })

with open("bikes_data.json", "w", encoding="utf-8") as f:

    json.dump(
        bike_json_data,
        f,
        ensure_ascii=False,
        indent=4
    )

print("\nBike data saved locally")

# =====================================================
# SPLIT DOCUMENTS
# =====================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)

docs = splitter.split_documents(documents)

# =====================================================
# EMBEDDINGS
# =====================================================

print("\nCreating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =====================================================
# VECTOR DATABASE
# =====================================================

vector_db = FAISS.from_documents(
    docs,
    embeddings
)

print("\nVector DB Ready")

# =====================================================
# DEEPSEEK LLM
# =====================================================

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0.8
)

# =====================================================
# SESSION STORAGE
# =====================================================

user_sessions = {}

# =====================================================
# REQUEST MODEL
# =====================================================

class ChatRequest(BaseModel):

    session_id: str
    message: str

# =====================================================
# FUZZY MATCH HELPERS
# =====================================================

def _normalize(text):
    """
    Lowercase, remove hyphens/underscores, collapse whitespace.
    We do NOT strip model code tokens here anymore — stripping
    'cb750' was causing 'CB 750 Hornet' to normalize to just
    'hornet', breaking matching entirely.
    Model-code-only noise (like 'gl1800' as a standalone token)
    is handled by the fuzzy scorer naturally.
    """
    text = text.lower()
    text = text.replace("-", " ").replace("_", " ")
    text = _re.sub(r'\s+', ' ', text).strip()
    return text


def _fuzzy(a, b):
    return SequenceMatcher(None, a, b).ratio()


def _keyword_overlap(extracted_norm, bike_norm):
    """
    What fraction of the bike's words appear in the extracted string?
    """
    bike_words = set(bike_norm.split())
    extracted_words = set(extracted_norm.split())

    if not bike_words:
        return 0.0

    overlap = bike_words.intersection(extracted_words)
    return len(overlap) / len(bike_words)


def _compute_score(extracted_raw, bike_name_raw):
    """
    Runs 6 strategies and returns the highest score.
    """
    ext_norm = _normalize(extracted_raw)
    bike_norm = _normalize(bike_name_raw)

    ext_raw = extracted_raw.lower().strip()
    bike_raw = bike_name_raw.lower().strip()

    s1 = _fuzzy(ext_norm, bike_norm)
    s2 = _fuzzy(ext_raw, bike_raw)
    s3 = _keyword_overlap(ext_norm, bike_norm)
    s4 = 1.0 if bike_norm and bike_norm in ext_norm else 0.0
    s5 = 1.0 if ext_norm and ext_norm in bike_norm else 0.0

    # s6: raw word overlap — fraction of bike's raw words found in extracted
    # This catches cases like extracted="cb 750 hornet", bike="CB 750 Hornet"
    # where normalization alone might miss short tokens
    bike_raw_words = [w for w in bike_raw.split() if len(w) >= 2]
    ext_raw_words  = set(ext_raw.split())
    s6 = (
        sum(1 for w in bike_raw_words if w in ext_raw_words) / len(bike_raw_words)
        if bike_raw_words else 0.0
    )

    best = max(s1, s2, s3, s4, s5, s6)

    print(
        f"  [{bike_name_raw}] "
        f"fuzzy_norm={s1:.2f} fuzzy_raw={s2:.2f} "
        f"keyword={s3:.2f} substr={max(s4,s5):.2f} "
        f"raw_word={s6:.2f} → {best:.2f}"
    )

    return best


def find_best_bike_match(extracted_bike, all_documents):
    """
    Searches ALL scraped documents using multi-strategy scoring.
    Returns: (bike_name, bike_image, bike_source) or ("", "", "")
    """

    best_score = 0.0
    best_doc = None

    print(f"\nMatching extracted: '{extracted_bike}'")

    for d in all_documents:

        current_bike = d.metadata.get("bike_name", "")

        if not current_bike or current_bike == "Unknown Bike":
            continue

        score = _compute_score(extracted_bike, current_bike)

        if score > best_score:
            best_score = score
            best_doc = d

    print(f"Best score: {best_score:.2f} → {best_doc.metadata.get('bike_name','') if best_doc else 'None'}")

    if best_doc and best_score > 0.3:
        return (
            best_doc.metadata.get("bike_name", ""),
            best_doc.metadata.get("image", ""),
            best_doc.metadata.get("source", "")
        )

    return ("", "", "")


def find_best_bike_match_from_json(extracted_bike):
    """
    Fallback: searches bikes_data.json using same multi-strategy scoring.
    Returns: (bike_name, bike_image, bike_source) or ("", "", "")
    """

    try:

        with open("bikes_data.json", "r", encoding="utf-8") as f:
            cached_bikes = json.load(f)

        best_score = 0.0
        best_bike = None

        print(f"\nJSON fallback matching: '{extracted_bike}'")

        for bike in cached_bikes:

            current_bike = bike.get("bike_name", "")

            if not current_bike or current_bike == "Unknown Bike":
                continue

            score = _compute_score(extracted_bike, current_bike)

            if score > best_score:
                best_score = score
                best_bike = bike

        print(f"JSON best score: {best_score:.2f} → {best_bike.get('bike_name','') if best_bike else 'None'}")

        if best_bike and best_score > 0.3:
            return (
                best_bike.get("bike_name", ""),
                best_bike.get("image", ""),
                best_bike.get("source", "")
            )

    except Exception as e:
        print("JSON fallback failed:", e)

    return ("", "", "")


# =====================================================
# NEGATIVE CONTEXT PHRASES
# =====================================================

NEGATIVE_CONTEXT_PHRASES = [
    "but it's", "but its", "but it is",
    "however it", "however,",
    "not really", "not the best", "not ideal",
    "wouldn't", "won't work", "doesn't suit",
    "too aggressive", "too extreme", "too much",
    "a commitment", "punishing", "punish your",
    "not happy", "not for", "cramped",
    "is hot", "it's hot",
    "would give you", "would be", "would feel",  # hypothetical framing
    "like the fireblade", "like the cbr",        # "like the X" = contrast bike
    "full-on", "full on",
    "in contrast", "on the other hand",
    "whereas", "while the",
    "compared to", "as opposed to",
]


def _is_bike_in_negative_context(bike_name, reply_text):
    """
    Checks whether the bike name appears ONLY in negative,
    dismissive, or hypothetical sentences in the reply.

    Strategy:
    1. Find ALL sentences containing the bike name.
    2. For each such sentence, check if it contains
       a negative context phrase.
    3. If EVERY sentence containing the bike is negative
       → the bike is being dismissed, return True (negative).
    4. If at least ONE sentence is positive/neutral
       → the bike is genuinely mentioned, return False.
    """

    if not bike_name or not reply_text:
        return False

    bike_norm_words = [
        w for w in _normalize(bike_name).split() if len(w) >= 2
    ]

    if not bike_norm_words:
        return False

    # Split reply into sentences
    sentences = _re.split(r'(?<=[.!?])\s+', reply_text)

    sentences_with_bike = []

    for sentence in sentences:

        sentence_norm = _normalize(sentence)

        if all(w in sentence_norm for w in bike_norm_words):
            sentences_with_bike.append(sentence.lower())

    if not sentences_with_bike:
        # Bike not found in any sentence
        return False

    # Check if each sentence containing the bike is negative
    negative_count = 0

    for sentence in sentences_with_bike:

        is_negative = any(
            phrase in sentence for phrase in NEGATIVE_CONTEXT_PHRASES
        )

        if is_negative:
            negative_count += 1

    # If ALL sentences with the bike are negative → truly dismissed
    if negative_count == len(sentences_with_bike):
        return True  # bike is in negative context only

    return False  # at least one positive/neutral mention


def is_bike_mentioned_positively(bike_name, reply_text):
    """
    Returns True if the bike name appears in the reply text
    AND is NOT used exclusively in a negative/dismissive context.

    Uses TWO matching strategies so short model codes like
    'cb', 'gl', 'nc' are not silently dropped:

    Strategy A — normalized word matching:
      Checks that all significant words (len >= 2) of the
      normalized bike name appear in the normalized reply.

    Strategy B — raw substring matching:
      Checks that the raw lowercased bike name (or its
      significant parts) appear as substrings in the reply.
      This catches "CB 750 Hornet" when the reply says
      "CB 750 Hornet" verbatim.

    Both strategies feed into the same negative-context check.
    """
    if not bike_name or not reply_text:
        return False

    bike_norm = _normalize(bike_name)
    reply_norm = _normalize(reply_text)
    reply_lower = reply_text.lower()

    # Strategy A: normalized word check (len >= 2 to include 'cb', 'gl' etc.)
    bike_words_norm = [w for w in bike_norm.split() if len(w) >= 2]

    # Strategy B: raw word check on lowercased strings
    bike_words_raw = [w for w in bike_name.lower().split() if len(w) >= 2]

    strategy_a = bool(bike_words_norm) and all(w in reply_norm for w in bike_words_norm)
    strategy_b = bool(bike_words_raw) and all(w in reply_lower for w in bike_words_raw)

    if not (strategy_a or strategy_b):
        return False  # not mentioned at all by either strategy

    # Is the mention only in negative/dismissive context?
    if _is_bike_in_negative_context(bike_name, reply_text):
        return False  # mentioned but dismissed

    return True  # mentioned positively


# =====================================================
# CREATE SESSION
# =====================================================

@app.get("/new_session")
def new_session():

    session_id = str(uuid.uuid4())

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    user_sessions[session_id] = memory

    return {
        "session_id": session_id
    }

# =====================================================
# CHAT ENDPOINT
# =====================================================

@app.post("/chat")
def chat(request: ChatRequest):

    # =========================================
    # CREATE SESSION IF NOT EXISTS
    # =========================================

    if request.session_id not in user_sessions:

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        user_sessions[request.session_id] = memory

    memory = user_sessions[request.session_id]

    # =========================================
    # CONVERSATION HISTORY
    # =========================================

    history_text = "\n".join([

        f"{msg.type}: {msg.content}"

        for msg in memory.chat_memory.messages

    ])

    # =========================================
    # RETRIEVE RELEVANT BIKES
    # =========================================

    retrieved_docs = vector_db.similarity_search(
        request.message,
        k=4
    )

    bike_context = "\n\n".join([

        d.page_content[:1500]

        for d in retrieved_docs

    ])

    # =========================================
    # BUILD DYNAMIC BIKE NAME LIST
    # =========================================

    available_bike_names = "\n".join([
        f"- {d.metadata.get('bike_name', '')}"
        for d in documents
        if d.metadata.get("bike_name")
    ])

    # =========================================
    # SYSTEM PROMPT
    # =========================================

    prompt = f"""
You are a premium AI motorcycle recommendation expert.

You behave like a real motorcycle enthusiast and riding consultant.

You are talking to ONE real user.

=================================================

CRITICAL BEHAVIOR RULES:

- Never say you are an AI
- Never sound robotic
- Never behave like a search engine
- Never dump specs unnecessarily
- Never summarize the entire database
- Never list too many bikes at once
- Never answer like documentation
- Never say:
    * "Based on provided information"
    * "According to context"
    * "I am an AI"

=================================================

YOUR GOAL:

Your job is to deeply understand:

- riding personality
- emotional taste
- riding style
- bike aesthetics preference
- comfort expectations
- sporty vs relaxed preference
- touring interest
- city vs highway usage
- adventure interest
- cruiser interest
- daily practicality
- beginner vs experienced
- emotional attraction toward motorcycles

=================================================

POSSIBLE BIKE CATEGORIES:

- Super Sport
- Touring
- Adventure
- Cruiser
- Street
- Off Road
- Scooter
- Commercial
- ATV

=================================================

IMPORTANT CONVERSATION RULES:

- Talk naturally like an experienced biker friend
- Keep responses conversational
- Ask ONE natural follow-up question at a time
- Guide the conversation slowly
- Avoid recommending too early
- Latest user message is MOST IMPORTANT
- Do not repeat previous answers
- Every response should move the conversation forward
- Sound emotionally intelligent

=================================================

RECOMMENDATION RULES:

- First understand the user deeply
- Infer the best category naturally
- Recommend ONLY when confidence is high
- When recommending a bike, ALWAYS mention the exact bike name clearly.
- Explain:
    * riding feel
    * comfort
    * emotional appeal
    * highway feel
    * city usability
    * premium feel
    * excitement level
    * practicality

- Compare bikes naturally when needed
- Recommendations must feel personalized

=================================================

FINAL OUTPUT RULE (ABSOLUTELY MANDATORY - NO EXCEPTIONS):

You MUST output the RECOMMENDED_BIKE tag ONLY when you are actively
recommending or suggesting a specific bike TO THE USER in your response.

Output the tag when:
1. You are confidently recommending a specific bike
2. You are leaning toward a bike and mentioning it by name to the user
3. You mentioned a specific bike as a strong positive candidate in your reply

DO NOT output the tag when:
- You are only asking questions and gathering information
- You have not yet mentioned any specific bike to the user
- You are still exploring the user's preferences without naming a bike

CRITICAL — NEVER tag a bike you mentioned negatively or dismissively:
- If you mentioned a bike ONLY to say it is NOT suitable → do NOT tag it
- If you used a bike as a contrast ("unlike the Fireblade...") → do NOT tag it
- If you described a bike as "too extreme", "too aggressive", "a commitment",
  "not ideal", "cramped", "punishing", "not really happy" → do NOT tag it
- The tag MUST only be the bike you are actively PUSHING to the user
- WRONG: "The Fireblade is exciting but too cramped. The CB750 Hornet is your bike."
  → RECOMMENDED_BIKE: CBR1000RR Fireblade  
- CORRECT: same response above
  → RECOMMENDED_BIKE: CB 750 Hornet  

ALWAYS end with this exact format when recommending:
RECOMMENDED_BIKE: exact bike name

Examples:
RECOMMENDED_BIKE: GL 1800 Goldwing Tour
RECOMMENDED_BIKE: CMX Rebel 1100
RECOMMENDED_BIKE: CB 750
RECOMMENDED_BIKE: CBR 650 R
RECOMMENDED_BIKE: CRF 1100 Adventure Sports

CRITICAL CONSISTENCY RULES:

RULE 1 - ALWAYS UPDATE THE TAG:
- Every response MUST have a fresh RECOMMENDED_BIKE tag if you positively named a bike
- If you changed your recommendation, the tag MUST reflect the NEW bike
- NEVER carry over the previous bike name — always output the bike you are currently recommending

RULE 2 - TAG MUST MATCH YOUR PRIMARY POSITIVE RECOMMENDATION:
- The RECOMMENDED_BIKE tag MUST be the bike you are pushing hardest and most positively
- If your closing sentence strongly favors one bike → RECOMMENDED_BIKE = that bike
- NEVER put a dismissed, contrast, or comparison bike in the tag

RULE 3 - MULTIPLE BIKES IN RESPONSE:
- If you compared two bikes, tag ONLY the one you ended up recommending
- The tag = the bike you want the user to actually consider buying
- The bike you used as a "but not this one" example → NEVER goes in the tag

RULE 4 - NO STALE TAGS:
- Ignore what the previous tag was
- Always decide the tag fresh based on THIS response only

AVAILABLE BIKE NAMES (use ONLY these exact names):
{available_bike_names}

IMPORTANT:
- ONLY recommend bikes from the list above
- The tag must use the exact name from the list above
- Do NOT add extra words like "Honda" before the name

=================================================

IMPORTANT RESPONSE STYLE:

- Short natural responses
- Maximum 2-4 paragraphs
- Avoid information overload
- Prefer follow-up questions
- Sound human and premium

=================================================

Conversation History:
{history_text}

=================================================

Relevant Honda Bike Information:
{bike_context}

=================================================

Latest User Message:
{request.message}

=================================================

If you still need more understanding:
ask smart follow-up questions BUT still output RECOMMENDED_BIKE if you mentioned any bike.

If user preferences are already clear:
give confident personalized recommendations and always output RECOMMENDED_BIKE.
"""

    # =========================================
    # GENERATE RESPONSE
    # =========================================

    response = llm.invoke(prompt)

    bot_reply = response.content

    # =========================================
    # DETECT RECOMMENDED BIKE
    # =========================================

    bike_name = ""
    bike_image = ""
    bike_source = ""

    # =========================================
    # FIND RECOMMENDED BIKE TAG
    # =========================================

    recommended_line = ""

    for line in bot_reply.splitlines():

        if "RECOMMENDED_BIKE:" in line:

            recommended_line = line.strip()

            break

    # =========================================
    # PROCESS TAG IF FOUND
    # =========================================

    if recommended_line:

        extracted_bike = recommended_line.replace(
            "RECOMMENDED_BIKE:",
            ""
        ).strip().lower()

        print(f"\nLLM tag extracted: '{extracted_bike}'")

        # =====================================
        # VALIDATE TAG — reject garbage values
        # =====================================

        def is_valid_bike_tag(text):
            if not text or len(text) < 3:
                return False
            junk_phrases = [
                "born to race", "feel the rush", "ride the future",
                "the legend", "born to win", "pure excitement",
                "the ultimate", "none", "n/a", "tbd", "unknown"
            ]
            if text.lower().strip() in junk_phrases:
                return False
            return True

        if not is_valid_bike_tag(extracted_bike):

            print(f"Extracted tag '{extracted_bike}' looks invalid, skipping match.")

        else:

            # Clean bot reply — remove the tag line before sending to frontend
            bot_reply = bot_reply.replace(recommended_line, "").strip()

            # =====================================
            # PRIMARY: Fuzzy match the LLM tag
            # against ALL scraped documents
            # =====================================

            bike_name, bike_image, bike_source = find_best_bike_match(
                extracted_bike,
                documents
            )

            # =====================================
            # FALLBACK: Search bikes_data.json
            # =====================================

            if not bike_name:

                print("Primary match failed, trying JSON fallback...")

                bike_name, bike_image, bike_source = find_best_bike_match_from_json(
                    extracted_bike
                )

            # =====================================
            # FIX: CROSS-VERIFY
            #
            # The LLM tag is the ground truth.
            # We only override if the tag bike is
            # completely ABSENT from the bot reply —
            # meaning it's a stale/hallucinated tag.
            #
            # We do NOT override based on text
            # position (old broken logic), because
            # the secondary bike often appears later
            # in the text even when the primary
            # recommendation is the tag bike.
            # =====================================

            if bike_name:

                tag_bike_positive = is_bike_mentioned_positively(bike_name, bot_reply)

                if not tag_bike_positive:

                    # Two sub-cases:
                    # A) Bike not mentioned at all → stale tag
                    # B) Bike mentioned only in negative/dismissive context
                    #    e.g. "the Fireblade is exciting but too cramped"
                    #    In both cases, scan the reply for the actual
                    #    positively-mentioned bike.

                    bike_norm_words = [
                        w for w in _normalize(bike_name).split() if len(w) >= 2
                    ]
                    reply_norm_check = _normalize(bot_reply)
                    reply_lower_check = bot_reply.lower()
                    physically_present = (
                        all(w in reply_norm_check for w in bike_norm_words)
                        or all(w in reply_lower_check for w in bike_name.lower().split() if len(w) >= 2)
                    ) if bike_norm_words else False

                    if physically_present:
                        print(
                            f"Tag bike '{bike_name}' found in reply but "
                            f"ONLY in negative/dismissive context → scanning for actual recommendation"
                        )
                    else:
                        print(
                            f"Tag bike '{bike_name}' NOT found in reply text → tag is stale"
                        )

                    print("Scanning reply text for positively-mentioned bike...")

                    best_text_score = 0.0
                    best_text_doc = None

                    for d in documents:

                        current_bike = d.metadata.get("bike_name", "")

                        if not current_bike:
                            continue

                        # Only consider bikes mentioned positively
                        if is_bike_mentioned_positively(current_bike, bot_reply):

                            bike_norm = _normalize(current_bike)
                            reply_norm = _normalize(bot_reply)
                            reply_lower_scan = bot_reply.lower()

                            # Use >= 2 so 'cb', 'gl', 'nc' etc. are included
                            bike_words_norm = [w for w in bike_norm.split() if len(w) >= 2]
                            bike_words_raw  = [w for w in current_bike.lower().split() if len(w) >= 2]

                            # Coverage: fraction of bike's words found in reply
                            # Try normalized first, fall back to raw
                            if bike_words_norm:
                                coverage = sum(
                                    1 for w in bike_words_norm if w in reply_norm
                                ) / len(bike_words_norm)
                            elif bike_words_raw:
                                coverage = sum(
                                    1 for w in bike_words_raw if w in reply_lower_scan
                                ) / len(bike_words_raw)
                            else:
                                coverage = 0.0

                            if coverage > best_text_score:
                                best_text_score = coverage
                                best_text_doc = d

                    if best_text_doc:
                        bike_name = best_text_doc.metadata.get("bike_name", "")
                        bike_image = best_text_doc.metadata.get("image", "")
                        bike_source = best_text_doc.metadata.get("source", "")
                        print(f"Corrected to positively-mentioned bike: '{bike_name}'")
                    else:
                        # No positively-mentioned bike found — clear result
                        print("No positively-mentioned bike found in reply — clearing result")
                        bike_name = ""
                        bike_image = ""
                        bike_source = ""

                else:

                    # Tag bike IS mentioned positively in reply → trust the LLM tag
                    print(f"Tag bike '{bike_name}' confirmed positively in reply → keeping LLM tag")

            # =====================================
            # LOG FINAL RESULT
            # =====================================

            if bike_name:
                print(f"\nFinal recommended bike: {bike_name}")
                print(f"Image: {bike_image}")
                print(f"Source: {bike_source}")
            else:
                print(f"\nNo match found for: '{extracted_bike}'")

    # =========================================
    # SAVE MEMORY
    # =========================================

    memory.chat_memory.add_user_message(
        request.message
    )

    memory.chat_memory.add_ai_message(
        bot_reply
    )

    # =========================================
    # RETURN RESPONSE
    # =========================================

    return {

        "session_id": request.session_id,

        "user": request.message,

        "bot": bot_reply,

        "recommended_bike": bike_name,

        "bike_image": bike_image,

        "bike_source": bike_source
    }

# =====================================================
# HOME
# =====================================================

@app.get("/")
def home():

    return {
        "status": "AI Bike Recommendation Bot Running"
    }

