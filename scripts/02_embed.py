import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "data", "arxiv_subset.parquet")
OUTPUT_FILE = os.path.join(BASE_DIR, "embeddings", "embeddings.npy")

# Створюємо директорію для ембеддингів, якщо вона не існує
os.makedirs(os.path.join(BASE_DIR, "embeddings"), exist_ok=True)

# Завантажуємо підготовлений датасет
df = pd.read_parquet(INPUT_FILE)

# Об'єднуємо title і abstract через [SEP]
texts = (df["title"] + " [SEP] " + df["abstract"]).tolist()

# Завантажуємо модель specter2_base з HuggingFace
model = SentenceTransformer("allenai/specter2_base")

# Кодуємо тексти в ембеддинги батчами по 64, нормалізуємо до одиничної довжини
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
)

# Виводимо результати
print(f"Загальна кількість текстів: {len(embeddings)}")
print(f"Розмірність ембеддингів: {embeddings.shape[1]}")
print(f"Норма першого ембеддингу: {np.linalg.norm(embeddings[0]):.6f}")

# Зберігаємо ембеддинги у форматі NumPy
np.save(OUTPUT_FILE, embeddings)
print(f"Збережено у {OUTPUT_FILE}")