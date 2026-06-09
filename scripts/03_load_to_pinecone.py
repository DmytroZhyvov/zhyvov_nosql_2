import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PARQUET = os.path.join(BASE_DIR, "data", "arxiv_subset.parquet")
INPUT_EMBEDDINGS = os.path.join(BASE_DIR, "embeddings", "embeddings.npy")

INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200

# Ініціалізація клієнта
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Створюємо індекс, якщо він ще не існує
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )
    print(f"Індекс '{INDEX_NAME}' створено")
else:
    print(f"Індекс '{INDEX_NAME}' вже існує")

# Підключаємось до індексу
index = pc.Index(INDEX_NAME)

# Завантажуємо датасет та ембеддинги
df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)

print(f"Завантажено {len(df)} записів і {len(embeddings)} ембеддингів")

# Завантажуємо дані
vectors = []
for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Підготовка батчів")):
    vectors.append({
        "id": f"paper_{i}",
        "values": embeddings[i].tolist(),
        "metadata": {
            "arxiv_id": row["id"],
            "title": row["title"],
            "abstract": row["abstract"][:500],
            "authors": row["authors"][:200],
            "year": int(row["year"]),
            "category": row["category"],
        },
    })

    # Відправляємо батч, коли накопичили BATCH_SIZE або дійшли до кінця
    if len(vectors) == BATCH_SIZE or i == len(df) - 1:
        index.upsert(vectors=vectors)
        vectors = []

# Виводимо кількість векторів в індексі
stats = index.describe_index_stats()
print(f"\nВсього векторів в індексі: {stats['total_vector_count']}")