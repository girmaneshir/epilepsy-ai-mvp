# test_semantic_search.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.translator import Translator
from src.embedder import Embedder
from src.vector_store import VectorStore

SAMPLES = [
    # --- English ---
    ("What is epilepsy?", "en"),
    ("What causes epilepsy?", "en"),
    ("What should I do if someone is having a seizure?", "en"),
    ("Can epilepsy be cured?", "en"),
    ("Is epilepsy contagious?", "en"),

    # --- Amharic (አማርኛ) ---
    ("ኤፒሊፕሲ ምንድነው?", "am"),
    ("የሚጥል በሽታ መንስኤዎች ምንድናቸው?", "am"),
    ("አንድ ሰው ሲንቀጠቀጥ ምን ማድረግ አለብኝ?", "am"),
    ("ኤፒሊፕሲ ይታከማል?", "am"),
    ("ኤፒሊፕሲ ተላላፊ ነው?", "am"),

    # --- Afaan Oromo ---
    ("Gaggabdoon maali?", "om"),
    ("Sababa gaggabdoo maali?", "om"),
    ("Yeroo nama tokko gaggabdu maal gochuu qabna?", "om"),
    ("Gaggabdoon wal'aansa qabaataa?", "om"),
    ("Gaggabdoon nama irraa namaatti darbaa?", "om"),

    # --- Tigrinya (ትግርኛ) ---
    ("ምንቅጥቃጥ እንታይ እዩ?", "ti"),
    ("ምንቅጥቃጥ ዝርዝር መግለጺ እንታይ እዩ?", "ti"),
    ("ሓደ ሰብ ክንቅጥቀጥ እንከሎ እንታይ ክገብር ኣለኒ?", "ti"),
    ("ምንቅጥቃጥ ይሕከም ድዩ?", "ti"),
    ("ምንቅጥቃጥ ካብ ሰብ ናብ ሰብ ይመሓላለፍ ድዩ?", "ti"),
]

def semantic_search(query: str, lang: str, translator, embedder, store):
    print(f"\n{'='*60}")
    print(f"Query ({lang}): {query}")

    # 1. Translate to English (if needed)
    if lang != "en":
        en_query = translator.translate(query, lang, "en")
        print(f"Translated EN:  {en_query}")
    else:
        en_query = query

    # 2. Embed the English query
    query_vector = embedder.embed([en_query])[0]

    # 3. Search ChromaDB
    results = store.query(query_vector, top_k=1) # Only get top 1 to save time
    
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    # 4. Print the top result
    if docs:
        score = 1 - distances[0]
        print(f"Top Match:      [{score:.4f}] from {metas[0]['source']}")
        
        # Take only the first 500 characters of the PDF text to save CPU time
        english_chunk = docs[0][:500]
        print(f"EN Content:     {english_chunk[:150].replace(chr(10), ' ')}...")

        # 5. BACK-TRANSLATE to Local Language
        if lang != "en":
            print(f"Translating back to {lang} (this takes ~5-10 seconds)...")
            local_chunk = translator.translate(english_chunk, "en", lang)
            print(f"Local Content:  {local_chunk[:200].replace(chr(10), ' ')}...")
    else:
        print("No matches found in ChromaDB.")

if __name__ == "__main__":
    print("Loading models... (This may take 30 seconds)")
    translator = Translator()
    embedder = Embedder()
    store = VectorStore()

    print(f"\nStarting Semantic Search Test with {len(SAMPLES)} samples...")
    print("WARNING: Back-translation is CPU-heavy. This will take 5-10 minutes.")
    
    for query, lang in SAMPLES:
        try:
            semantic_search(query, lang, translator, embedder, store)
        except Exception as e:
            print(f"\n[ERROR] Failed on '{query}' ({lang}): {e}")

    print(f"\n{'='*60}")
    print("Semantic search test complete.")