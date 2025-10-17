# backend/app/utils/build_index.py
import os, json, numpy as np
from tqdm import tqdm
from app.utils.preprocess import align_and_crop
from app.models.embedder import load_embedder, get_embedding
import faiss
from PIL import Image

DATA_ROOT = "datasets"  # put celeb/, cartoon/, animal/ here
OUT_DIR = "app/static/faiss"
os.makedirs(OUT_DIR, exist_ok=True)

def build():
    model = load_embedder()
    embeddings = []
    metadata = {}
    idx = 0

    # Walk through all subfolders
    for domain in os.listdir(DATA_ROOT):
        domain_path = os.path.join(DATA_ROOT, domain)
        if not os.path.isdir(domain_path):
            continue

        # Go through subfolders like "person1", "person2"
        for subdir, _, files in os.walk(domain_path):
            for fname in tqdm(files, desc=f"Processing {subdir}"):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                p = os.path.join(subdir, fname)
                try:
                    img = Image.open(p).convert("RGB")
                    face = align_and_crop(img)
                    if face is None:
                        continue
                    emb = get_embedding(model, face)
                    embeddings.append(emb)
                    metadata[str(idx)] = {"path": f"/{p}", "domain": domain, "filename": fname}
                    idx += 1
                except Exception as e:
                    print("skip", p, e)

    if not embeddings:
        print("⚠️ No embeddings generated — make sure there are valid face images.")
        return

    embs = np.stack(embeddings).astype('float32')
    d = embs.shape[1]
    nlist = 100
    index = faiss.index_factory(d, f"IVF{nlist},Flat")
    print("Training FAISS index...")
    index.train(embs)
    index.add(embs)
    faiss.write_index(index, os.path.join(OUT_DIR, "faceworld.index"))

    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Index built successfully with {len(embeddings)} embeddings.")
    print(f"Saved to: {OUT_DIR}")

if __name__ == "__main__":
    build()

