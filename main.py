# === main.py ===

import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from google.cloud import vision
import openai
import numpy as np

# === 環境変数のロード ===
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcloud-key.json"
openai.api_key = os.getenv("OPENAI_API_KEY")

# === FastAPI アプリケーション初期化 ===
app = FastAPI()

# === Vision API クライアント初期化 ===
vision_client = vision.ImageAnnotatorClient()

# === 既知の著名IPのテキスト例 ===
KNOWN_IP_TEXTS = [
    "Mickey Mouse by Disney",
    "Naruto by Masashi Kishimoto",
    "Pikachu from Pokémon",
    "Frozen Elsa Disney Princess",
    "Hello Kitty",
    "Attack on Titan",
    "Minions by Universal",
    "Studio Ghibli Totoro"
]

# === ロゴ検出（Google Vision）===
def detect_logos_google(image_path):
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = vision_client.logo_detection(image=image)
    return [logo.description for logo in response.logo_annotations]

# === 埋め込み生成（OpenAI）===
def get_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response['data'][0]['embedding']

# === コサイン類似度計算 ===
def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# === /upload エンドポイント ===
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    os.makedirs("temp", exist_ok=True)
    file_path = f"temp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # ロゴ検出
    logos = detect_logos_google(file_path)

    # 埋め込み & 類似度スコア計算
    user_embedding = get_embedding(file.filename)
    known_embeddings = [get_embedding(text) for text in KNOWN_IP_TEXTS]
    scores = [cosine_similarity(user_embedding, emb) for emb in known_embeddings]
    max_score = max(scores)

    # 判定ロジック
    if logos:
        result = {
            "is_official": True,
            "reason": "Google Visionでロゴが検出されました",
            "logos": logos
        }
    elif max_score > 0.9:
        result = {
            "is_official": True,
            "reason": "OpenAIの埋め込みが著名IPに非常に類似しています",
            "similarity": max_score
        }
    else:
        result = {
            "is_official": False,
            "reason": "類似するIPが検出されませんでした",
            "similarity": max_score
        }

    return JSONResponse(content=result)
