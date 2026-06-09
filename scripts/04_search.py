import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

# Підключаємось до Pinecone та завантажуємо модель
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)

# Завантажуємо датасет для повного abstract
df = pd.read_parquet(os.path.join(BASE_DIR, "data", "arxiv_subset.parquet"))

def encode_query(query: str) -> list:
    """Кодуємо запит в ембеддинг"""
    return model.encode(query, normalize_embeddings=True).tolist()

def print_results(results, header: str):
    print(f"\n{'='*60}")
    print(f" {header}")
    print(f"{'='*60}")
    for i, match in enumerate(results.matches, 1):
        meta = match.metadata
        print(f"\n#{i} | score: {match.score:.4f}")
        print(f"  Назва:    {meta['title']}")
        print(f"  Категорія:{meta['category']} | Рік: {meta['year']}")
        print(f"  Абстракт: {meta['abstract'][:150]}...")

# Виконуємо чистий семантичний пошук
query = "teaching machines to recognize objects in pictures"
query_vector = encode_query(query)

results = index.query(vector=query_vector, top_k=TOP_K, include_metadata=True)
print_results(results, f"Семантичний пошук: '{query}'")


# Виконуємо пошук з фільтрацією ─

# Приклад A: reinforcement learning, категорія cs.LG
query_a = "reinforcement learning"
vector_a = encode_query(query_a)

results_a = index.query(
    vector=vector_a,
    top_k=TOP_K,
    include_metadata=True,
    filter={"category": {"$eq": "cs.LG"}},
)
print_results(results_a, "Фільтр A: reinforcement learning | category=cs.LG")

# Приклад B: старі статті до 2015 року
results_b = index.query(
    vector=vector_a,
    top_k=TOP_K,
    include_metadata=True,
    filter={"year": {"$lt": 2015}},
)
print_results(results_b, "Фільтр B: reinforcement learning | year < 2015")


# Порівнюємо метрики схожості локально
embeddings = np.load(os.path.join(BASE_DIR, "embeddings", "embeddings.npy"))
query_vec = np.array(encode_query(query))


def print_local_results(indices, scores, header):
    print(f"\n{'='*60}")
    print(f" {header}")
    print(f"{'='*60}")
    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        row = df.iloc[idx]
        print(f"\n#{rank} | score: {score:.4f}")
        print(f"  Назва:    {row['title']}")
        print(f"  Категорія:{row['category']} | Рік: {row['year']}")


# Cosine similarity (для нормалізованих = dot product)
cosine_scores = embeddings @ query_vec
top_cosine = np.argsort(cosine_scores)[::-1][:TOP_K]
print_local_results(top_cosine, cosine_scores[top_cosine], "Локально: Cosine Similarity")

# Dot product
dot_scores = embeddings @ query_vec
top_dot = np.argsort(dot_scores)[::-1][:TOP_K]
print_local_results(top_dot, dot_scores[top_dot], "Локально: Dot Product")

# L2 distance (менше = краще)
l2_scores = np.linalg.norm(embeddings - query_vec, axis=1)
top_l2 = np.argsort(l2_scores)[:TOP_K]
print_local_results(top_l2, l2_scores[top_l2], "Локально: L2 Distance (менше = краще)")