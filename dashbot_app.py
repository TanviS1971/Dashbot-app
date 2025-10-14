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
    # Normalize craving to match Chroma naming (no spaces, lowercase)
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
        # Run fetch script
        result = subprocess.run(
            ["python3", "fetch_serpapi_data.py", zip_code, craving or ""],
            capture_output=True,
            text=True,
            timeout=90
        )
        
        if result.returncode != 0:
            print(f"❌ Fetch failed: {result.stderr}")
            return False
        
        # Run build script
        result = subprocess.run(
            ["python3", "build_store.py", zip_code],
            capture_output=True,
            text=True,
            timeout=90
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

    # Get or create collection
    collection = get_collection_for_zip(zip_code, craving)
    if not collection:
        print(f"📥 No data for {zip_code}, fetching now...")
        craving = normalize_craving(craving)
        success = fetch_and_build_for_zip(zip_code, craving)
        if not success:
            return [{
            "name": "🍔 Uh oh!",
            "categories": "System Notice",
            "rating": "N/A",
            "address": "",
            "zip_code": zip_code
        }]
        collection = get_collection_for_zip(zip_code, craving)

    if not collection:
        return []

    try:
        # Build search query
        if neighborhood:
            search_query = f"{craving} {neighborhood} {zip_code}"
        else:
            search_query = f"{craving} {zip_code}"

        print(f"🔍 Searching: {search_query}")
        user_vector = embedding_model.encode(search_query).tolist()

        results = collection.query(
            query_embeddings=[user_vector],
            n_results=30,
            include=["metadatas"]
        )

        restaurants = results.get("metadatas", [[]])[0]
        print(f"📊 Found {len(restaurants)} restaurants")

        # Exclude previously shown
        restaurants = [r for r in restaurants if r.get("name") not in exclude_names]

        # Sort by rating
        restaurants = sorted(
            restaurants,
            key=lambda r: float(r.get("rating", 0)) if r.get("rating") != "N/A" else 0,
            reverse=True
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
# 💬 CONVERSATIONAL RESPONSE
# ==============================
def generate_response(user_input, restaurants, session_state):
    """Generate warm, dynamic, emotion-aware responses with graceful endings."""
    # ====== Tone & Sentiment Detection ======
    angry_words = ["angry", "upset", "frustrated", "mad", "annoyed", "pissed", "irritated"]
    happy_words = ["thank you", "thanks", "perfect", "yay", "awesome", "great", "love it", "ok thank you", "thankyou", "ok thanks"]
    goodbye_words = ["bye", "see you", "thanks bye", "thank you bye", "goodnight", "good night", "take care", "see ya"]

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

    # ====== Handle Emotional Responses ======
    if tone == "frustrated":
        return (
            f"I’m really sorry you’re feeling frustrated, {session_state.name} 😔 "
            "Let’s fix this! Maybe tell me exactly what kind of food you’re craving or your nearby street name? "
            "I’ll do my best to find something you’ll love ❤️"
        )

    # 💬 Graceful chat ending for 'thank you', 'ok thanks', 'bye', etc.
    if tone == "grateful" and ("thank" in user_lower or "bye" in user_lower or "ok" in user_lower):
        return (
            f"Aww, thank you {session_state.name}! 💖 I'm so glad I could help today. "
            "I'll remember your craving in case you come back later! "
            "Enjoy your meal and see you soon 🍜✨"
        )
    
    # ===== No Restaurants Found =====
    if not restaurants:
    # Check for Google Places daily limit (fetch_serpapi_data.py returns False on quota hit)
        if restaurants is False:
            return "🍔 Uh oh! I’ve hit my daily limit for restaurant lookups — come back tomorrow for fresh foodie picks! ❤️"

        if tone == "grateful":
            return f"You're very welcome, {session_state.name}! 🍽️ Hope your next meal is amazing!"
        else:
            return (
            f"Looks like I couldn’t find any great {session_state.last_craving or 'food'} spots "
            f"around {session_state.zip_code} 😅 Maybe try a nearby ZIP or tweak the craving?"
            )


    # ====== Restaurant Recommendations ======
    restaurant_info = []
    for i, r in enumerate(restaurants, 1):
        name = r.get("name")
        rating = r.get("rating")
        cats = r.get("categories")
        restaurant_info.append(f"{i}. **{name}** ({cats}) - {rating}⭐")

    restaurants_text = "\n".join(restaurant_info)

    # ====== Dynamic Prompt for LLM ======
    system_prompt = f"""
    You are DashBot — a warm, playful, emotionally intelligent foodie assistant 🍜.
    - Respond in a natural, human tone that matches the user's emotion.
    - Be concise, conversational, and kind.
    - If the user is happy or thankful, respond cheerfully.
    - If they sound neutral, stay engaging and warm.
    - If frustrated, be gentle and apologetic.

    CRITICAL RULES:
    - Never make up restaurants.
    - Always present exactly 3 from the list below.
    - Always stay context-aware (name, ZIP, craving).

    User: {session_state.name}
    ZIP: {session_state.zip_code}
    Detected Tone: {tone.upper()}

    Restaurants:
    {restaurants_text}
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=0.75,
            max_tokens=400,
        )

        reply = response.choices[0].message.content.strip()

        if tone == "grateful":
            reply += "\n\n💖 So glad I could help! Come back anytime you’re craving something delicious!"
        else:
            reply += "\n\n💡 *You can find these easily on DoorDash!*"

        return reply

    except Exception as e:
         error_message = str(e).lower()
    if "rate limit" in error_message or "quota" in error_message or "limit exceeded" in error_message:
        return (
            "🍔 Oops! DashBot’s foodie brain is out of chats for the day. "
            "Come back tomorrow for more delicious discoveries! 😋"
        )
    else:
        print(f"⚠️ LLM error: {e}")
        simple = f"Here are 3 {session_state.last_craving or 'great'} spots, {session_state.name}! 😋\n\n"
        for i, r in enumerate(restaurants, 1):
            simple += f"{i}. **{r.get('name')}** ({r.get('categories')}) ⭐ {r.get('rating')}\n"
        simple += "\n💡 *Try searching these on DoorDash!*"
        return simple


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
        if any(word in user_input.lower() for word in ["help", "find", "don't know", "do not know", "not sure", "unknown", "no idea"]):
            return (
                "No worries! 💌 Here’s how to find it:\n"
                "🔍 Google 'what is my zip code'\n"
                "📱 Or visit: https://www.zip-codes.com/search.asp\n"
                "Then tell me your 5-digit ZIP!"
            )

        zip_match = re.search(r"\b\d{5}\b", user_input)
        if zip_match:
            session_state.zip_code = zip_match.group(0)
            session_state.stage = "neighborhood"
            return (
                f"Perfect! ZIP {session_state.zip_code} 📍\n\n"
                "What neighborhood are you in? (e.g., Downtown, Capitol Hill)\n"
                "Or type 'skip' for the whole area!"
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

        # --- Detect move/new location ---
        moved_phrases = [
            "moved", "new city", "new place", "different area",
            "changed location", "relocated", "i’m in", "i am in"
        ]
        if any(phrase in user_text for phrase in moved_phrases):
            session_state.stage = "zip"
            return "No worries! Let’s update your location 🏡 What’s your new ZIP code?"

        # --- Detect ZIP change mid-conversation ---
        zip_match = re.search(r"\b\d{5}\b", user_input)
        if zip_match:
            new_zip = zip_match.group(0)
            if new_zip != session_state.zip_code:
                session_state.zip_code = new_zip
                return f"Got it! Switched to ZIP {new_zip} 📍 What are you craving?"

        # --- Handle 'more' (get new recommendations) ---
        if any(word in user_text for word in ["more", "another", "else"]):
            if session_state.last_craving and session_state.last_restaurants:
                exclude = [r.get("name") for r in session_state.last_restaurants]
                restaurants = search_restaurants(
                    session_state.last_craving,
                    session_state.zip_code,
                    getattr(session_state, "neighborhood", ""),
                    exclude_names=exclude
                )
                session_state.last_restaurants = restaurants
                return generate_response(session_state.last_craving, restaurants, session_state)
            return "Sure! What kind of food are you in the mood for? 😊"

        # --- Handle 'order' or 'menu' ---
        if any(word in user_text for word in ["order", "menu", "link"]):
            if session_state.last_restaurants:
                top = session_state.last_restaurants[0]
                name = top.get("name")
                return f"Ready to order from **{name}**? 🍽️\n\nSearch '{name}' on the DoorDash app!"
            return "Tell me what you're craving first 😄"

        # --- Main craving search ---
        craving = user_input
        session_state.last_craving = craving

        restaurants = search_restaurants(
            craving,
            session_state.zip_code,
            getattr(session_state, "neighborhood", "")
        )
        session_state.last_restaurants = restaurants

        return generate_response(user_input, restaurants, session_state)

    return "I'm here to help you find delicious food! 🍜"
