# backend/app/utils/build_index.py
import os, json, numpy as np
from tqdm import tqdm
from PIL import Image
from insightface.app import FaceAnalysis
import faiss
from PIL import Image


DATA_ROOT = "datasets"  # Folder containing face folders
DATA_ROOT = "datasets"
OUT_DIR = "app/static/faiss"
os.makedirs(OUT_DIR, exist_ok=True)

def build():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0)
    embeddings = []
    metadata = {}
    idx = 0


    # ✅ Walk through ALL subdirectories and files
    for root, dirs, files in os.walk(DATA_ROOT):
        print(f"Scanning: {root}")
        for fname in files:  # <-- only iterate over files, not directories

    # Walk through all subfolders
    for root, dirs, files in os.walk(DATA_ROOT):
        print(f"📂 Scanning: {root}")
        for fname in tqdm(files, desc=f"Processing {os.path.basename(root)}"):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            p = os.path.join(root, fname)


            try:
                img = Image.open(p).convert("RGB")
                face = align_and_crop(img)
                if face is None:
                    print(f"⚠️ No face detected in {p}")

            try:
                img = np.array(Image.open(p).convert("RGB"))
                faces = app.get(img)

                if not faces:
                    print(f"❌ Skip {p}: No face detected")
                    continue

                # Take the first detected face
                emb = faces[0].normed_embedding
                embeddings.append(emb)




                metadata[str(idx)] = {
                    "path": f"/{p}",
                    "domain": os.path.basename(os.path.dirname(p)),
                    "filename": fname
                }
                idx += 1

            except Exception as e:
                print("skip", p, e)

    if not embeddings:
        print("⚠️ No embeddings generated — make sure there are valid face images.")
                print(f"⚠️ Error processing {p}: {e}")

    if not embeddings:
        print("⚠️ No embeddings generated — make sure valid face images are present.")
        return

    # Build FAISS index
    embs = np.stack(embeddings).astype('float32')
    d = embs.shape[1]
    nlist = 100
    index = faiss.index_factory(d, f"IVF{nlist},Flat")
    print("Training FAISS index...")
    index.train(embs)
    index = faiss.IndexFlatL2(d)
    index.add(embs)
    faiss.write_index(index, os.path.join(OUT_DIR, "faceworld.index"))

    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Index built successfully with {len(embeddings)} embeddings.")
    print(f"Saved to: {OUT_DIR}")

if __name__ == "__main__":
    build()
