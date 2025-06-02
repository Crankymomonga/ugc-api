# === main.py ===

import os
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
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

# === 既知の著名IPテキスト（埋め込み比較用）===
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

# === ロゴ検出（Google Cloud Vision）===
def detect_logos_google(image_path):
    try:
        with open(image_path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = vision_client.logo_detection(image=image)
        return [logo.description for logo in response.logo_annotations]
    except Exception as e:
        print("Vision API Error:", e)
        return []

# === 埋め込み生成（OpenAI）===
def get_embedding(text):
    try:
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        return response['data'][0]['embedding']
    except Exception as e:
        print("OpenAI Embedding Error:", e)
        return None

# === コサイン類似度計算 ===
def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

# === /upload エンドポイント ===
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # ファイル保存
    os.makedirs("temp", exist_ok=True)
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    temp_filename = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join("temp", temp_filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # ロゴ検出
    logos = detect_logos_google(file_path)

    # OpenAI埋め込み & 類似度スコア計算
    user_embedding = get_embedding(file.filename)  # filename が altテキストの代替になる想定
    if user_embedding is None:
        raise HTTPException(status_code=500, detail="Failed to get embedding from OpenAI")

    known_embeddings = [get_embedding(text) for text in KNOWN_IP_TEXTS]
    scores = [cosine_similarity(user_embedding, emb) for emb in known_embeddings if emb]
    max_score = max(scores) if scores else 0.0

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

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 10000))  # Render環境用
    uvicorn.run("main:app", host="0.0.0.0", port=port)
