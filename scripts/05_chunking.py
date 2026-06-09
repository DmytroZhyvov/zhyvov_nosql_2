import os
import re
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
BATCH_SIZE = 100

# Параметри chunking
FIXED_CHUNK_SIZE = 100   # слів
FIXED_OVERLAP = 20       # слів перекриття
SEMANTIC_MAX_WORDS = 100 # максимум слів у семантичному чанку

INDEX_FIXED = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet(os.path.join(BASE_DIR, "data", "arxiv_subset.parquet"))


# Вибираємо 30 статей з найдовшими анотаціями
df["abstract_len"] = df["abstract"].str.split().str.len()
top30 = df.nlargest(30, "abstract_len").reset_index(drop=True)
print(f"Обрано {len(top30)} статей, макс. довжина: {top30['abstract_len'].max()} слів")


# Стратегія 1: Fixed-size chunking
def fixed_chunking(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Розбиваємо текст на чанки фіксованого розміру з перекриттям"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# Стратегія 2: Semantic chunking
def semantic_chunking(text: str, max_words: int) -> list[str]:
    """Об'єднуємо речення до досягнення ліміту слів та зберігаємо цілі речення"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        if current_len + len(words) > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.extend(words)
        current_len += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


# Створюємо індекси у Pinecone
def create_index(name: str):
    if name not in pc.list_indexes().names():
        pc.create_index(
            name=name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Індекс '{name}' створено")
    else:
        print(f"Індекс '{name}' вже існує")
    return pc.Index(name)


index_fixed = create_index(INDEX_FIXED)
index_semantic = create_index(INDEX_SEMANTIC)


# Готуємо та завантажуємо чанки
def prepare_and_upload(index, rows: pd.DataFrame, strategy: str):
    vectors = []
    chunk_counter = 0

    for _, row in tqdm(rows.iterrows(), total=len(rows), desc=f"Chunking [{strategy}]"):
        text = row["title"] + " [SEP] " + row["abstract"]

        if strategy == "fixed":
            chunks = fixed_chunking(text, FIXED_CHUNK_SIZE, FIXED_OVERLAP)
        else:
            chunks = semantic_chunking(text, SEMANTIC_MAX_WORDS)

        for chunk_num, chunk_text in enumerate(chunks):
            embedding = model.encode(chunk_text, normalize_embeddings=True).tolist()

            vectors.append({
                "id": f"{strategy}_{row['id']}_chunk{chunk_num}",
                "values": embedding,
                "metadata": {
                    "arxiv_id": row["id"],
                    "title": row["title"],
                    "chunk_text": chunk_text[:500],
                    "chunk_num": chunk_num,
                    "year": int(row["year"]),
                    "category": row["category"],
                },
            })
            chunk_counter += 1

            # Завантажуємо батч
            if len(vectors) >= BATCH_SIZE:
                index.upsert(vectors=vectors)
                vectors = []

    # Завантажуємо залишок
    if vectors:
        index.upsert(vectors=vectors)

    print(f"[{strategy}] Завантажено {chunk_counter} чанків")


prepare_and_upload(index_fixed, top30, "fixed")
prepare_and_upload(index_semantic, top30, "semantic")


# Шукаємо по чанкам
def search_chunks(query: str, index, label: str):
    query_vec = model.encode(query, normalize_embeddings=True).tolist()
    results = index.query(vector=query_vec, top_k=5, include_metadata=True)

    print(f"\n{'='*60}")
    print(f" [{label}] Запит: '{query}'")
    print(f"{'='*60}")
    for i, match in enumerate(results.matches, 1):
        meta = match.metadata
        print(f"\n#{i} | score: {match.score:.4f}")
        print(f"  Стаття:  {meta['title']}")
        print(f"  Чанк #{meta['chunk_num']}: {meta['chunk_text'][:200]}...")


test_queries = [
    "quantum computing and entanglement",
    "machine learning optimization methods",
]

for query in test_queries:
    search_chunks(query, index_fixed, "Fixed")
    search_chunks(query, index_semantic, "Semantic")