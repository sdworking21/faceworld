# backend/app/main.py
import io, os, base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from app.utils.preprocess import align_and_crop
from app.models.embedder import get_embedding, load_embedder
from app.utils.faiss_indexer import FaissIndex
from app.models.generator import Generator
from PIL import Image

app = FastAPI(title="FaceWorld API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend origin for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# mount outputs for static serving
os.makedirs("app/static/outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="app/static/outputs"), name="outputs")

# Singletons (load once)
embedder = load_embedder()          # ArcFace / InsightFace wrapper
faiss_index = FaissIndex("app/static/faiss/index_path")  # will load index files
generator = Generator(model_dir="app/models/checkpoints")  # GAN/Diffusion wrapper

@app.post("/preview")
async def preview(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    # align and crop (returns PIL Image)
    cropped = align_and_crop(image)
    if cropped is None:
        return JSONResponse({"error": "no_face_detected"}, status_code=400)
    out_path = "app/static/outputs/preview.jpg"
    cropped.save(out_path)
    return {"preview_url": f"/outputs/preview.jpg"}

@app.post("/search")
async def search(file: UploadFile = File(...), domain: str = Form("celebrity"), top_k: int = Form(3)):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    cropped = align_and_crop(image)
    if cropped is None:
        return JSONResponse({"error": "no_face_detected"}, status_code=400)
    emb = get_embedding(embedder, cropped)  # numpy vector
    hits = faiss_index.search(emb, top_k=top_k, domain=domain)
    # hits -> list of {id, score, path}
    return {"results": hits}

@app.post("/generate")
async def generate(file: UploadFile = File(...), style: str = Form("cartoon")):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    cropped = align_and_crop(image)
    if cropped is None:
        return JSONResponse({"error": "no_face_detected"}, status_code=400)
    generated = generator.generate(cropped, style=style)  # PIL image or ndarray
    # Save result
    out_name = f"out_{style}.jpg"
    out_path = os.path.join("app/static/outputs", out_name)
    generated.save(out_path)
    return {"generated_url": f"/outputs/{out_name}"}
