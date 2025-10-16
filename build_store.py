import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import sys
import glob
import re  

def normalize_craving(craving):
    """Normalize craving text consistently across scripts."""
    if not craving:
        return ""
    craving = craving.lower().strip()
    craving = re.sub(r"[^a-z0-9]+", "_", craving)
    craving = craving.strip("_")
    return craving


def build_vector_store(zip_code=None):
    """Build vector store for specific ZIP or auto-detect from CSV."""
    
    print("=" * 60)
    print("DashBot Vector Store Builder (Per-ZIP)")
    print("=" * 60)
    
    # ==================================================
    # FIND CSV FILE
    # ==================================================
    if zip_code:
        candidates = glob.glob(f"restaurants_{zip_code}*.csv")
        if not candidates:
            raise FileNotFoundError(f"No CSV found for ZIP {zip_code}")
        csv_path = max(candidates, key=os.path.getmtime)
    else:
        csv_files = glob.glob("restaurants_*.csv")
        if not csv_files:
            csv_path = "restaurants.csv"
        else:
            csv_path = max(csv_files, key=os.path.getmtime)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found!")
    
    print(f"Loading {csv_path}")
    
    # ==================================================
    # Extract craving name from filename
    # ==================================================
    craving = None
    match = re.search(r"restaurants_(\d+)_?(.*)\.csv", os.path.basename(csv_path))
    if match:
        detected_zip = match.group(1)
        craving_raw = match.group(2).strip()
        craving = craving_raw if craving_raw else None
    else:
        detected_zip = zip_code or "generic"

    # Normalize craving (so “Indian food” → “indian_food”)
    safe_craving = normalize_craving(craving) if craving else "general"
    collection_name = f"restaurants_{detected_zip}{'_' + safe_craving if safe_craving else ''}"
    
    print(f"Collection to build: {collection_name}")
    
    # ==================================================
    # LOAD CSV DATA
    # ==================================================
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise Exception(f"Error reading CSV: {e}")
    
    print("\n Cleaning data...")
    initial_count = len(df)
    df = df.dropna(subset=["name"])
    df = df.drop_duplicates(subset=["name", "address"])
    df = df.reset_index(drop=True)
    
    removed = initial_count - len(df)
    if removed > 0:
        print(f"   Removed {removed} invalid/duplicate entries")
    
    print(f" {len(df)} unique restaurants ready")
    if len(df) == 0:
        raise ValueError(" No valid restaurants!")
    
    # ==================================================
    # LOAD EMBEDDING MODEL
    # ==================================================
    print("\n Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(" Model loaded successfully")
    
    # ==================================================
    #  CONNECT TO CHROMA DB
    # ==================================================
    print("\n Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_data")
    collection = client.get_or_create_collection(collection_name)
    print(f" Connected to collection '{collection_name}'")
    
    # Clear existing entries
    try:
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
            print(" Cleared previous data from collection")
    except Exception:
        pass
    
    # ==================================================
    #  GENERATE EMBEDDINGS
    # ==================================================
    print("\n Preparing embeddings...")
    texts = df["embedding_text"].fillna("").tolist()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    print(f" Generated {len(embeddings)} embeddings")
    
    # ==================================================
    #  STORE IN CHROMA
    # ==================================================
    print("\n Storing embeddings in ChromaDB...")
    ids = [str(i) for i in range(len(df))]
    metadatas = [
        {
            "name": str(row.get("name", "")),
            "categories": str(row.get("categories", "")),
            "rating": str(row.get("rating", "")),
            "address": str(row.get("address", "")),
            "zip_code": str(row.get("zip_code", "")),
        }
        for _, row in df.iterrows()
    ]
    
    batch_size = 100
    for i in tqdm(range(0, len(ids), batch_size), desc="Indexing"):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            metadatas=metadatas[i:end],
            documents=texts[i:end],
        )
    
    print(f" Indexed {len(df)} restaurants")
    
    final_count = collection.count()
    print(f"\n Verification: {final_count} items now in collection")
    print("\n" + "=" * 60)
    print(" Build Complete!")
    print("=" * 60)
    print(f" Collection: {collection_name}")
    print(f" Restaurants: {final_count}")
    print(f" Chroma Data Path: ./chroma_data/")
    print("=" * 60)


# ==================================================
#  MAIN ENTRY POINT
# ==================================================
if __name__ == "__main__":
    try:
        zip_code = sys.argv[1] if len(sys.argv) > 1 else None
        build_vector_store(zip_code)
    except KeyboardInterrupt:
        print("\n\n Interrupted")
    except Exception as e:
        print(f"\n Failed: {e}")
        sys.exit(1)
