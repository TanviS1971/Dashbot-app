import chromadb
import os

def check_collection():
    """Check ChromaDB collection status and display sample data"""
    
    print("=" * 60)
    print("🔍 DashBot Collection Inspector")
    print("=" * 60)
    
    # Check if chroma_data directory exists
    if not os.path.exists("./chroma_data"):
        print("❌ chroma_data directory not found!")
        print("\n💡 Solution: Run 'python build_store.py' to create the vector store")
        return
    
    try:
        # Connect to ChromaDB
        client = chromadb.PersistentClient(path="./chroma_data")
        collection = client.get_or_create_collection("dashbot_restaurants")
        
        print("✅ Connected to Chroma!")
        
        # Get collection count
        count = collection.count()
        print(f"📦 Total items in collection: {count}")
        
        if count == 0:
            print("\n⚠️ The collection is empty!")
            print("\n💡 Solution:")
            print("   1. Run: python fetch_serpapi_data.py")
            print("   2. Then: python build_store.py")
        else:
            # Display sample metadata
            print("\n🔍 Sample restaurant entries:")
            print("-" * 60)
            
            sample = collection.get(limit=5)
            
            for i, meta in enumerate(sample["metadatas"], 1):
                name = meta.get('name', 'Unknown')
                categories = meta.get('categories', 'N/A')
                rating = meta.get('rating', 'N/A')
                address = meta.get('address', 'N/A')
                zip_code = meta.get('zip_code', 'N/A')
                
                print(f"\n{i}. {name}")
                print(f"   📍 {address}")
                print(f"   🏷️  {categories}")
                print(f"   ⭐ Rating: {rating}")
                print(f"   📮 ZIP: {zip_code}")
            
            # Statistics
            print("\n" + "=" * 60)
            print("📊 Collection Statistics:")
            print("=" * 60)
            
            # Count restaurants with ZIP codes
            all_data = collection.get()
            zip_count = sum(1 for meta in all_data["metadatas"] if meta.get("zip_code"))
            
            print(f"✅ Total restaurants: {count}")
            print(f"📮 With ZIP codes: {zip_count}")
            print(f"❓ Without ZIP codes: {count - zip_count}")
            
            # Get unique categories (sample)
            categories_sample = set()
            for meta in sample["metadatas"]:
                cats = meta.get('categories', '')
                if cats:
                    categories_sample.update(cats.split(', ')[:3])
            
            if categories_sample:
                print(f"\n🏷️  Sample categories: {', '.join(list(categories_sample)[:5])}")
            
            print("\n✅ Vector store is ready for use!")
            print("💡 Run: streamlit run streamlit_app.py")
    
    except Exception as e:
        print(f"❌ Error connecting to ChromaDB: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure you've run: python build_store.py")
        print("   2. Check if ./chroma_data directory exists")
        print("   3. Try deleting ./chroma_data and rebuilding")

if __name__ == "__main__":
    check_collection()