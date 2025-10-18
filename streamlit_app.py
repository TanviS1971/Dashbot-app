import streamlit as st
import re
import pandas as pd
from dashbot_app import dashbot_reply

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="🍔 DashBot", page_icon="🍜", layout="centered")

# ==============================
# DEBUG SIDEBAR
# ==============================
st.sidebar.markdown("### 🔍 Debug Info")
st.sidebar.markdown(f"**App loaded at:** {pd.Timestamp.now()}")
st.sidebar.markdown(f"**Python running:** ✅")

# ==============================
# CUSTOM CSS
# ==============================
st.markdown("""
<style>
    .stApp {
        background-color: #FFF8F6;
        font-family: 'Segoe UI', sans-serif;
    }
    .user-bubble {
        background-color: #FFD9D0;
        color: #000;
        padding: 12px 18px;
        border-radius: 20px 20px 2px 20px;
        margin: 8px 0;
        max-width: 75%;
        float: right;
        clear: both;
        word-wrap: break-word;
        border: 1px solid #F4B1A3;
    }
    .bot-bubble {
        background-color: #FFFFFF;
        color: #000;
        padding: 12px 18px;
        border-radius: 20px 20px 20px 2px;
        margin: 8px 0;
        max-width: 75%;
        float: left;
        clear: both;
        word-wrap: break-word;
        border: 1px solid #E4E4E4;
    }
    .bot-header {
        text-align: center;
        margin-bottom: 20px;
    }
    .bot-header img {
        width: 100px;
        height: 100px;
        border-radius: 50%;
    }
    .bot-header h2 {
        margin-top: 10px;
        font-size: 1.5rem;
        color: #333;
    }
    @media (max-width: 768px) {
        .user-bubble, .bot-bubble {
            max-width: 85%;
            font-size: 0.9rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# SESSION STATE INIT
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hey! I'm Dash, your personal DoorDash food finder! I'll help you discover amazing restaurants near you. What's your name?"}
    ]
    st.session_state.stage = "name"
    st.session_state.name = ""
    st.session_state.zip_code = ""
    st.session_state.neighborhood = ""
    st.session_state.last_craving = None
    st.session_state.last_restaurants = []

# ==============================
# DEBUG SIDEBAR - Session State
# ==============================
if st.sidebar.checkbox("Show Session State", value=True):
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Current Session:**")
    st.sidebar.write(f"🏷️ Stage: `{st.session_state.stage}`")
    st.sidebar.write(f"👤 Name: `{st.session_state.name or 'Not set'}`")
    st.sidebar.write(f"📍 ZIP: `{st.session_state.zip_code or 'Not set'}`")
    st.sidebar.write(f"🏘️ Neighborhood: `{st.session_state.neighborhood or 'Not set'}`")
    st.sidebar.write(f"🍽️ Craving: `{st.session_state.last_craving or 'Not set'}`")
    st.sidebar.write(f"📊 Restaurants found: `{len(st.session_state.last_restaurants)}`")
    
    if st.session_state.last_restaurants:
        st.sidebar.markdown("**Last Search Results:**")
        for i, r in enumerate(st.session_state.last_restaurants, 1):
            st.sidebar.write(f"{i}. {r.get('name')} ⭐ {r.get('rating')}")

# ==============================
# (Optional) CACHE HOOK
# ==============================
@st.cache_data(show_spinner=False)
def cache_restaurants(zip_code, craving):
    """Stub function for caching fetched data (extend later if needed)."""
    return f"Cached dataset for {zip_code}-{craving}"

# ==============================
# HEADER
# ==============================
st.markdown("""
<div class="bot-header">
    <img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png" alt="DashBot">
    <h2>DashBot — Your DoorDash Assistant</h2>
</div>
""", unsafe_allow_html=True)

# ==============================
# DISPLAY CHAT
# ==============================
for chat in st.session_state.messages:
    bubble_class = "user-bubble" if chat["role"] == "user" else "bot-bubble"
    st.markdown(f'<div class="{bubble_class}">{chat["content"]}</div>', unsafe_allow_html=True)

# --- Auto-scroll to bottom after rendering ---
st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)

# ==============================
# CHAT INPUT
# ==============================
placeholder_map = {
    "name": "Type your name...",
    "zip": "Enter your 5-digit ZIP code...",
    "neighborhood": "e.g., Capitol Hill, Downtown...",
    "craving": "What are you craving? 🍕🍜🍣"
}

user_input = st.chat_input(placeholder_map.get(st.session_state.stage, "Type here..."))

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Show loading spinner during data fetching
    with st.spinner("🍳 Cooking up your restaurant list..."):
        try:
            reply = dashbot_reply(user_input, st.session_state)
        except Exception as e:
            reply = f"⚠️ Oops! Something went wrong: {str(e)}\n\nPlease try again!"
            st.error(f"Error details: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Add bot reply
    st.session_state.messages.append({"role": "assistant", "content": reply})
    
    st.rerun()

# ==============================
# DEBUG EXPANDER
# ==============================
with st.expander("🔍 Detailed Debug Panel", expanded=False):
    st.markdown("### Session State Details")
    st.json({
        "stage": st.session_state.stage,
        "name": st.session_state.name,
        "zip_code": st.session_state.zip_code,
        "neighborhood": st.session_state.neighborhood,
        "last_craving": st.session_state.last_craving,
        "restaurants_count": len(st.session_state.last_restaurants),
        "messages_count": len(st.session_state.messages)
    })
    
    if st.session_state.last_restaurants:
        st.markdown("### Last Restaurant Search Results")
        st.dataframe(pd.DataFrame(st.session_state.last_restaurants))

# ==============================
# START OVER BUTTON
# ==============================
# Only show when user is not in name stage
if st.session_state.stage != "name":
    if st.button("🔄 Start Over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ==============================
# FOOTER DEBUG INFO
# ==============================
st.markdown("---")
st.caption("🤖 DashBot v1.0 | Powered by Google Places API & Groq LLM")
if st.session_state.stage == "craving" and st.session_state.last_restaurants:
    st.caption(f"✅ Found {len(st.session_state.last_restaurants)} restaurants in ZIP {st.session_state.zip_code}")