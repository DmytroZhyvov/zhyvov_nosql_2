import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet(os.path.join(BASE_DIR, "data", "arxiv_subset.parquet")).reset_index(drop=True)


# Будуємо BM25-індекс
# Токенізуємо title + abstract для кожної статті
corpus = (df["title"] + " " + df["abstract"]).tolist()
tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)
print(f"BM25 індекс побудовано для {len(corpus)} документів")


# BM25 пошук
def search_bm25(query: str, top_k: int = TOP_K) -> list[dict]:
    """Повертаємо топ-K результатів BM25 з індексом і score"""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {"idx": int(i), "score": float(scores[i]), "title": df.iloc[i]["title"],
         "category": df.iloc[i]["category"], "year": df.iloc[i]["year"]}
        for i in top_indices
    ]


# Векторний пошук (Pinecone)
def search_vector(query: str, top_k: int = TOP_K) -> list[dict]:
    """Повертаємо топ-K результатів векторного пошуку з Pinecone"""
    query_vec = model.encode(query, normalize_embeddings=True).tolist()
    results = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
    return [
        {"id": m.id, "score": float(m.score), "title": m.metadata["title"],
         "category": m.metadata["category"], "year": m.metadata["year"]}
        for m in results.matches
    ]


# Reciprocal Rank Fusion
def rrf(bm25_results: list[dict], vector_results: list[dict], k: int = 60) -> list[dict]:
    """
    RRF формула: score(d) = Σ 1 / (k + rank(d))
    k=60 — стандартне значення, що згладжує вплив топових позицій
    """
    rrf_scores = {}

    # Додаємо BM25 ранги
    for rank, result in enumerate(bm25_results, start=1):
        doc_id = str(result["idx"])
        rrf_scores[doc_id] = rrf_scores.get(doc_id, {
            "title": result["title"],
            "category": result["category"],
            "year": result["year"],
            "rrf_score": 0.0,
        })
        rrf_scores[doc_id]["rrf_score"] += 1.0 / (k + rank)

    # Додаємо векторні ранги
    for rank, result in enumerate(vector_results, start=1):
        # Pinecone повертає id вигляду "paper_123"
        doc_id = result["id"].replace("paper_", "")
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "title": result["title"],
                "category": result["category"],
                "year": result["year"],
                "rrf_score": 0.0,
            }
        rrf_scores[doc_id]["rrf_score"] += 1.0 / (k + rank)

    # Сортуємо за RRF score
    sorted_results = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return sorted_results[:TOP_K]


# Виводимо результати
def print_results(results: list[dict], header: str, score_key: str = "score"):
    print(f"\n{'='*60}")
    print(f" {header}")
    print(f"{'='*60}")
    for i, r in enumerate(results[:5], 1):
        score = r.get(score_key, r.get("rrf_score", 0))
        print(f"#{i} | score: {score:.4f} | {r['category']} {int(r['year'])}")
        print(f"     {r['title']}")


# Демонстрація трьох запитів
queries = [
    "BERT fine-tuning",
    "Yann LeCun convolutional networks",
    "making computers understand human emotions from text",
]

for query in queries:
    print(f"\n\n{'#'*60}")
    print(f"  ЗАПИТ: {query}")
    print(f"{'#'*60}")

    bm25_res = search_bm25(query)
    vector_res = search_vector(query)
    hybrid_res = rrf(bm25_res, vector_res)

    print_results(bm25_res, "BM25")
    print_results(vector_res, "Векторний пошук (Pinecone)")
    print_results(hybrid_res, "Гібридний пошук (RRF)", score_key="rrf_score")