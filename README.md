Introducing DashBot!
=
A conversational AI assistant that helps you discover restaurants based on your location and cravings. Was Designed as a potential new feature idea to reduce decision fatigue on Doordash

What It Does
=
->Asks for your name, ZIP code, and what you're craving

->Fetches restaurant data from Google Places API
->Provides personalized restaurant recommendations through conversation


Data Sources
=
Google Places API - Real-time restaurant data including names, ratings, categories, and addresses
Location Data - Uses ZIP code geocoding to find restaurants within a 5km radius
Restaurant Metadata - Captures ratings, cuisine types, and addresses for semantic matching

LLM & AI Components
=
Groq API — Powers the conversational responses and natural language understanding
SentenceTransformers — Generates semantic embeddings using the all-MiniLM-L6-v2 model
ChromaDB — Vector database for semantic search of restaurant descriptions


Tech Stack
=
Frontend: Streamlit
APIs: Google Places API, Groq API
Vector Database: ChromaDB
Embeddings: SentenceTransformers (all-MiniLM-L6-v2)
Data Processing: Pandas

Project Structure
dashbot/
├── fetch_serpapi_data.py       # Fetches restaurant data from Google Places API
├── build_vector_store.py      # Builds ChromaDB vector store from restaurant data
├── dashbot_app.py             # Core chatbot logic and conversation handling
├── streamlit_app.py           # Streamlit UI interface
├── .env                       # Environment variables
├── chroma_data/              # Local vector database storage
└── restaurants_*.csv         # Temporary restaurant data files

Privacy & Data Storage
=
DashBot does not store any personal user data.

No user information is saved to databases
Conversation history exists only in the active session
Restaurant data is fetched fresh from Google Places API each time
ZIP codes and names are used only during the session and are not logged
All data is cleared when you close the app or click "Start Over"
Local CSV and vector files contain only public restaurant information from Google Places

