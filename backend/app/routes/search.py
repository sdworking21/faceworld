import json
import faiss
import numpy as np
from fastapi import APIRouter, UploadFile, Form, HTTPException
from app.models.embedder import embedder
from app.config import STATIC_DIR
from app.utils.image_utils import read_image
import os
from datetime import datetime

router = APIRouter()

# Paths to FAISS index and metadata
FAISS_INDEX_PATH = os.path.join(STATIC_DIR, "faiss", "faceworld.index")
METADATA_PATH = os.path.join(STATIC_DIR, "faiss", "metadata.json")

# Load FAISS index + metadata on startup
if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    print(f"✅ Loaded FAISS index with {index.ntotal} embeddings")
else:
    print("⚠️ No FAISS index found. Run `python -m app.utils.build_index` first.")
    index, metadata = None, {}

@router.post("/search")
async def search(
    file: UploadFile,
    domain: str = Form("celebrity"),
    top_k: int = Form(3)
):
    """Search similar faces from FAISS index"""
    if index is None or index.ntotal == 0:
        raise HTTPException(status_code=400, detail="FAISS index not loaded or empty")

    # Read image bytes
    try:
        image_bytes = await file.read()
        img = read_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Get face embedding
    faces = embedder.get(img)
    if len(faces) == 0:
        raise HTTPException(status_code=404, detail="No face detected in the image")

    emb = np.array(faces[0].embedding, dtype=np.float32).reshape(1, -1)

    # 🔍 Perform FAISS search
    try:
        D, I = index.search(emb, top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAISS search failed: {str(e)}")

    # Collect results
    results = []
    for i, dist in zip(I[0], D[0]):
        key = str(i)
        if key in metadata:
            entry = metadata[key]
            results.append({
                "id": key,
                "person_name": entry.get("person", "Unknown"),
                "image_path": entry.get("path", ""),
                "distance": float(dist)
            })
        else:
            results.append({
                "id": key,
                "person_name": "Unknown",
                "image_path": "",
                "distance": float(dist)
            })

    # Save query image for reference
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_path = os.path.join(STATIC_DIR, "outputs", f"query_{ts}.jpg")
    os.makedirs(os.path.dirname(query_path), exist_ok=True)
    with open(query_path, "wb") as f:
        f.write(image_bytes)

    # ✅ Debug information for you
    print("=== DEBUG /search ===")
    print(f"Embedding shape: {emb.shape}")
    print(f"FAISS index count: {index.ntotal}")
    print(f"Top-K Results: {results}")
    print("=====================")

    return {"query_image": f"/outputs/query_{ts}.jpg", "results": results}
