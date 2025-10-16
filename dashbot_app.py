import os
import re
import random
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq
import subprocess

# ==============================
# 🌟 ENV SETUP
# ==============================
load_dotenv()

try:
    import streamlit as st
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
except:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found!")

client = Groq(api_key=GROQ_API_KEY)

# ==============================
# 🧠 MODELS & DATABASE
# ==============================
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_data")


def normalize_craving(craving):
    """Normalize craving text consistently across scripts."""
    if not craving:
        return ""
    craving = craving.lower().strip()
    craving = re.sub(r"[^a-z0-9]+", "_", craving)
    craving = craving.strip("_")
    return craving


def get_collection_for_zip(zip_code, craving=None):
    """Get collection for specific ZIP + craving (if exists)."""
    safe_craving = normalize_craving(craving)
    collection_name = f"restaurants_{zip_code}{'_' + safe_craving if safe_craving else ''}"
    print(f"🧭 Looking for collection: {collection_name}")

    try:
        collection = chroma_client.get_collection(collection_name)
        count = collection.count()
        if count > 0:
            print(f"✅ Found {count} restaurants in {collection_name}")
            return collection
        else:
            print(f"⚠️ Collection {collection_name} is empty.")
            return None
    except Exception as e:
        print(f"⚠️ Could not load {collection_name}: {e}")
        return None


def fetch_and_build_for_zip(zip_code, craving=None):
    """Fetch restaurants and build vector store for ZIP + craving."""
    print(f"🔄 Fetching data for ZIP {zip_code} (craving: {craving or 'general'})...")

    try:
        result = subprocess.run(
            ["python3", "fetch_serpapi_data.py", zip_code, craving or ""],
            capture_output=True,
            text=True,
            timeout=90,
        )
        if result.returncode != 0:
            print(f"❌ Fetch failed: {result.stderr}")
            return False

        result = subprocess.run(
            ["python3", "build_store.py", zip_code],
            capture_output=True,
            text=True,
            timeout=90,
        )
        if result.returncode != 0:
            print(f"❌ Build failed: {result.stderr}")
            return False

        print(f"✅ Successfully built collection for {zip_code}")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


# ==============================
# 🍱 RESTAURANT SEARCH
# ==============================
def search_restaurants(craving, zip_code, neighborhood=None, exclude_names=None):
    """Search for top 3 restaurants (excludes previous ones if specified)."""
    exclude_names = exclude_names or []

    craving = craving.lower().strip()
    craving = re.sub(
        r"\b(hot|spicy|tasty|delicious|yummy|good|nice|warm|fresh|best|real|authentic)\b",
        "",
        craving,
        flags=re.I,
    ).strip()

    synonym_map = {
        "ramen noodles": "ramen",
        "noodles": "ramen",
        "shawarma wrap": "shawarma",
        "indian curry": "indian",
        "mexican tacos": "mexican",
    }
    craving = synonym_map.get(craving, craving)

    collection = get_collection_for_zip(zip_code, craving)
    if not collection:
        print(f"📥 No data for {zip_code}, fetching now...")
        craving = normalize_craving(craving)
        success = fetch_and_build_for_zip(zip_code, craving)
        if not success:
            return [
                {
                    "name": "🍔 Uh oh!",
                    "categories": "System Notice",
                    "rating": "N/A",
                    "address": "",
                    "zip_code": zip_code,
                }
            ]
        collection = get_collection_for_zip(zip_code, craving)

    if not collection:
        return []

    try:
        search_query = f"{craving} {neighborhood or ''} {zip_code}"
        print(f"🔍 Searching: {search_query}")
        user_vector = embedding_model.encode(search_query).tolist()

        results = collection.query(
            query_embeddings=[user_vector], n_results=30, include=["metadatas"]
        )

        restaurants = results.get("metadatas", [[]])[0]
        print(f"📊 Found {len(restaurants)} restaurants")

        restaurants = [r for r in restaurants if r.get("name") not in exclude_names]

        restaurants = sorted(
            restaurants,
            key=lambda r: float(r.get("rating", 0))
            if r.get("rating") != "N/A"
            else 0,
            reverse=True,
        )

        top3 = restaurants[:3]
        print(f"✅ Returning top {len(top3)}:")
        for r in top3:
            print(f"   - {r.get('name')} ({r.get('categories')}) ⭐ {r.get('rating')}")
        return top3

    except Exception as e:
        print(f"⚠️ Search error: {e}")
        return []


# ==============================
# 💬 CONVERSATIONAL RESPONSE (updated, grounded)
# ==============================
def generate_response(user_input, restaurants, session_state):
    """Generate warm, dynamic, emotion-aware responses with grounded factual data."""
    # ====== Tone Detection ======
    angry_words = ["angry", "upset", "frustrated", "mad", "annoyed", "pissed"]
    happy_words = ["thank you", "thanks", "perfect", "awesome", "great", "love it", "ok thanks"]
    goodbye_words = ["bye", "see you", "goodnight", "take care"]

    user_lower = user_input.lower().strip()
    tone = "neutral"
    end_convo = False

    if any(w in user_lower for w in angry_words):
        tone = "frustrated"
    elif any(w in user_lower for w in happy_words):
        tone = "grateful"
    elif any(w in user_lower for w in goodbye_words):
        tone = "grateful"
        end_convo = True

    if tone == "frustrated":
        return (
            f"I’m really sorry you’re feeling frustrated, {session_state.name} 😔 "
            "Let’s fix this! Maybe tell me exactly what kind of food you’re craving or your nearby street name? ❤️"
        )

    if tone == "grateful" and ("thank" in user_lower or "bye" in user_lower):
        return (
            f"Aww, thank you {session_state.name}! 💖 I'm so glad I could help. "
            "Enjoy your meal and see you soon 🍜✨"
        )

    # ===== No Restaurants Found =====
    if not restaurants:
        print("⚠️ DEBUG — No restaurants returned from search.")
        return (
            f"Looks like I couldn’t find any {session_state.last_craving or 'food'} spots "
            f"around {session_state.zip_code} 😅 Maybe try another ZIP or craving?"
        )

    # ====== Build restaurant context ======
    restaurant_context = "\n".join(
        [
            f"{i+1}. {r['name']} — rated {r['rating']}⭐. "
            f"Located at {r['address']}. Category: {r['categories']}. "
            f"Perfect for cravings like {session_state.last_craving or 'great food'}."
            for i, r in enumerate(restaurants)
        ]
    )

    # ====== Grounded system prompt ======
    system_prompt = f"""
You are DashBot 🍜, a friendly but factual restaurant recommender.

CRITICAL RULES:
- Only use restaurants listed below. These are trusted database results.
- Never invent or rename any restaurant.
- If the list seems empty, say you couldn’t find any results.
- Keep tone warm, concise, and conversational.
- End by asking which one sounds best.
- Be sure to add address and ratings for restaurants you mention.

USER:
Name: {session_state.name}
ZIP: {session_state.zip_code}
Craving: {session_state.last_craving or 'food'}

RESTAURANTS (trusted data):
{restaurant_context}
"""

    try:
        print(f"DEBUG — Sending {len(restaurants)} restaurants to LLM.")
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=0,  # deterministic mode
            top_p=1,
            max_tokens=400,
        )

        reply = response.choices[0].message.content.strip()
        reply += "\n\n💡 *You can find these easily on DoorDash!*"
        return reply

    except Exception as e:
        print(f"⚠️ LLM error: {e}")
        fallback = f"Here are {len(restaurants)} {session_state.last_craving or 'great'} spots, {session_state.name}! 😋\n\n"
        for i, r in enumerate(restaurants, 1):
            fallback += f"{i}. **{r['name']}** ({r['categories']}) ⭐ {r['rating']}\n"
        fallback += "\n💡 *Try searching these on DoorDash!*"
        return fallback


# ==============================
# 💬 MAIN CHAT LOGIC
# ==============================
def dashbot_reply(user_input, session_state):
    """Main conversational flow."""
    if not hasattr(session_state, "last_restaurants"):
        session_state.last_restaurants = []
    if not hasattr(session_state, "last_craving"):
        session_state.last_craving = None

    # === NAME STAGE ===
    if session_state.stage == "name":
        cleaned = user_input.lower()
        for phrase in ["my name is", "i am", "i'm", "call me"]:
            cleaned = cleaned.replace(phrase, "")
        name = cleaned.strip().split()[-1].capitalize() if cleaned.strip() else "Friend"
        if not name.isalpha():
            name = "Friend"
        session_state.name = name
        session_state.stage = "zip"
        return f"Nice to meet you, {name}! 🥰 What's your ZIP code?"

    # === ZIP STAGE ===
    elif session_state.stage == "zip":
        if any(word in user_input.lower() for word in ["help", "find", "don't know", "unknown"]):
            return (
                "No worries! 💌 Here’s how to find it:\n"
                "🔍 Google 'what is my zip code'\n📱 Or visit: https://www.zip-codes.com/search.asp\nThen tell me your 5-digit ZIP!"
            )

        zip_match = re.search(r"\b\d{5}\b", user_input)
        if zip_match:
            session_state.zip_code = zip_match.group(0)
            session_state.stage = "neighborhood"
            return (
                f"Perfect! ZIP {session_state.zip_code} 📍\n\n"
                "What neighborhood are you in? (e.g., Downtown, Capitol Hill)\nOr type 'skip' for the whole area!"
            )
        else:
            return "Hmm, that doesn't look valid 🤔 Try entering a 5-digit ZIP (like 98105)."

    # === NEIGHBORHOOD STAGE ===
    elif session_state.stage == "neighborhood":
        if "skip" in user_input.lower():
            session_state.neighborhood = ""
            session_state.stage = "craving"
            return f"No problem! I’ll look all around {session_state.zip_code} 🍽️ What are you craving?"
        else:
            session_state.neighborhood = user_input.strip()
            session_state.stage = "craving"
            return f"Awesome! Searching near {session_state.neighborhood} 🎯 What are you craving?"

    # === CRAVING STAGE ===
    elif session_state.stage == "craving":
        user_text = user_input.lower().strip()

        moved_phrases = ["moved", "new city", "different area", "relocated", "i’m in"]
        if any(p in user_text for p in moved_phrases):
            session_state.stage = "zip"
            return "No worries! Let’s update your location 🏡 What’s your new ZIP code?"

        zip_match = re.search(r"\b\d{5}\b", user_input)
        if zip_match:
            new_zip = zip_match.group(0)
            if new_zip != session_state.zip_code:
                session_state.zip_code = new_zip
                return f"Got it! Switched to ZIP {new_zip} 📍 What are you craving?"

        if any(word in user_text for word in ["more", "another", "else"]):
            if session_state.last_craving and session_state.last_restaurants:
                exclude = [r.get("name") for r in session_state.last_restaurants]
                restaurants = search_restaurants(
                    session_state.last_craving,
                    session_state.zip_code,
                    getattr(session_state, "neighborhood", ""),
                    exclude_names=exclude,
                )
                session_state.last_restaurants = restaurants
                return generate_response(session_state.last_craving, restaurants, session_state)
            return "Sure! What kind of food are you in the mood for? 😊"

        if any(word in user_text for word in ["first", "second", "third", "this one", "that one", "sounds good", "perfect"]):
            if session_state.last_restaurants:
                chosen = None
                if "first" in user_text:
                    chosen = session_state.last_restaurants[0]
                elif "second" in user_text and len(session_state.last_restaurants) > 1:
                    chosen = session_state.last_restaurants[1]
                elif "third" in user_text and len(session_state.last_restaurants) > 2:
                    chosen = session_state.last_restaurants[2]
                else:
                    chosen = session_state.last_restaurants[0]
                name = chosen.get("name")
                return (
                    f"Yay, {session_state.name}! 🎉 Great choice — {name} is a local favorite. "
                    f"Go ahead and check them out on DoorDash 🍕✨"
                )

        if any(word in user_text for word in ["order", "menu", "link"]):
            if session_state.last_restaurants:
                top = session_state.last_restaurants[0]
                name = top.get("name")
                return f"Ready to order from **{name}**? 🍽️\n\nSearch '{name}' on the DoorDash app!"
            return "Tell me what you're craving first 😄"

        craving = user_input
        session_state.last_craving = craving

        restaurants = search_restaurants(
            craving, session_state.zip_code, getattr(session_state, "neighborhood", "")
        )
        session_state.last_restaurants = restaurants

        return generate_response(user_input, restaurants, session_state)

    return "I'm here to help you find delicious food! 🍜"

