import requests
import pandas as pd
import os
import re
import sys
from dotenv import load_dotenv
import time

# ===== LOAD ENV =====
load_dotenv()

try:
    import streamlit as st
    API_KEY = st.secrets.get("GOOGLE_PLACES_API_KEY")
except:
    API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

if not API_KEY:
    raise ValueError("❌ GOOGLE_PLACES_API_KEY not found!")

# ==============================
# 📍 UTILITIES
# ==============================

def normalize_craving(craving):
    """Normalize craving text consistently across scripts."""
    if not craving:
        return ""
    craving = craving.lower().strip()
    craving = re.sub(r"[^a-z0-9]+", "_", craving)
    craving = craving.strip("_")
    return craving

def extract_zip(address):
    """Extract 5-digit ZIP from address."""
    match = re.search(r"\b\d{5}(?:-\d{4})?\b", address or "")
    return match.group(0) if match else ""

def validate_zip_code(zip_code):
    """Validate 5-digit ZIP."""
    return bool(re.match(r"^\d{5}(-\d{4})?$", zip_code))

# ==============================
# 🍽️ FETCH RESTAURANTS
# ==============================
def fetch_restaurants(zip_code, craving=None):
    """
    Fetch restaurants for specific ZIP, optionally filtered by craving (e.g., 'indian', 'mexican').
    Saves results as normalized filename: restaurants_<zip>_<craving>.csv
    """
    craving_text = f" for craving '{craving}'" if craving else ""
    print(f"🍽️ Fetching restaurants for ZIP {zip_code}{craving_text}...")
    
    if not validate_zip_code(zip_code):
        print(f"❌ Invalid ZIP: {zip_code}")
        return False
    
    # --- Geocode the ZIP to get lat/lng ---
    geo_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={zip_code}&key={API_KEY}"
    
    try:
        geo_res = requests.get(geo_url, timeout=10).json()
    except Exception as e:
        print(f"❌ Network error: {e}")
        return False
    
    if not geo_res.get("results"):
        print(f"❌ Could not geocode ZIP {zip_code}")
        return False
    
    location = geo_res["results"][0]["geometry"]["location"]
    lat, lng = location["lat"], location["lng"]
    print(f"📍 Location: ({lat}, {lng})")
    
    # --- Fetch restaurants ---
    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": 5000,             # 5 km search radius
        "rankby": "prominence",     # rank by rating/popularity instead of distance
        "type": "restaurant",
        "keyword": craving or "",   # user's craving keyword
        "key": API_KEY
    }
    
    all_restaurants = []
    page = 0
    
    while page < 3:  # Max 3 pages (≈60 restaurants)
        page += 1
        print(f"   🔎 Fetching page {page}...")
        
        try:
            res = requests.get(base_url, params=params, timeout=10)
            data = res.json()
        except Exception as e:
            print(f"❌ Request error: {e}")
            break
        
        if data.get("status") == "OVER_QUERY_LIMIT":
            print("⚠️ Google Places API quota exceeded!")
            print("🍔 Sorry! DashBot’s map explorer has reached its daily limit.")
            return False
        elif data.get("status") not in ["OK", "ZERO_RESULTS"]:
            print(f"⚠️ API Error: {data.get('status')}")
            break
        
        results = data.get("results", [])
        if not results:
            break
        
        for r in results:
            name = r.get("name", "")
            rating = r.get("rating", "N/A")
            
            # Clean categories
            types = r.get("types", [])
            meaningful = [t for t in types if t not in [
                "point_of_interest", "establishment", "food", "restaurant"
            ]]
            categories = ", ".join([t.replace("_", " ").title() for t in meaningful[:3]])
            if not categories:
                categories = "Restaurant"
            
            address = r.get("vicinity", "")
            extracted_zip = extract_zip(address) or zip_code
            
            if not name:
                continue
            
            embedding_text = (
                               f"{name}. Category: {categories}. "
                               f"Known for {categories} dishes. Located at {address}. "
                               f"Rated {rating} stars. A popular spot for people craving {categories} or similar foods."
                              )
            
            all_restaurants.append({
                "name": name,
                "rating": rating,
                "categories": categories,
                "address": address,
                "zip_code": extracted_zip,
                "embedding_text": embedding_text
            })
        
        print(f"   ✓ Added {len(results)} results")
        
        # Pagination
        next_token = data.get("next_page_token")
        if not next_token:
            break
        params["pagetoken"] = next_token
        time.sleep(2)
    
    if not all_restaurants:
        print("❌ No restaurants found")
        return False
    
    # --- Filter top by rating ---
    df = pd.DataFrame(all_restaurants)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
    df.drop_duplicates(subset=["name", "address"], inplace=True)
    df = df.sort_values(by="rating", ascending=False)
    
    # --- Normalize craving for consistent filenames ---
    safe_craving = normalize_craving(craving) if craving else "general"
    output_path = f"restaurants_{zip_code}_{safe_craving}.csv"
    
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✅ Saved {len(df)} restaurants to {output_path}")
    return True

# ==============================
# 🚀 MAIN
# ==============================
if __name__ == "__main__":
    if len(sys.argv) > 2:
        zip_code = sys.argv[1]
        craving = sys.argv[2]
    elif len(sys.argv) > 1:
        zip_code = sys.argv[1]
        craving = None
    else:
        zip_code = input("Enter ZIP code: ").strip()
        craving = input("Enter craving (e.g. 'indian', 'mexican', 'sushi'): ").strip() or None
    
    success = fetch_restaurants(zip_code, craving)
    sys.exit(0 if success else 1)
