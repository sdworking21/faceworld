# backend/app/utils/build_index.py
import os, json, numpy as np
from tqdm import tqdm
from app.utils.preprocess import align_and_crop
from app.models.embedder import load_embedder, get_embedding
import faiss

DATA_ROOT = "datasets"  # put celeb/, cartoon/, animal/ here
OUT_DIR = "app/static/faiss"
os.makedirs(OUT_DIR, exist_ok=True)

def build():
    model = load_embedder()
    embeddings = []
    metadata = {}
    idx = 0
    for domain in os.listdir(DATA_ROOT):
        domain_path = os.path.join(DATA_ROOT, domain)
        if not os.path.isdir(domain_path): continue
        for fname in tqdm(os.listdir(domain_path)):
            p = os.path.join(domain_path, fname)
            try:
                from PIL import Image
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
        print("no embeddings generated")
        return
    embs = np.stack(embeddings).astype('float32')
    # choose IVF index for scale
    d = embs.shape[1]
    nlist = 100  # tune for dataset; more images -> larger nlist
    index = faiss.index_factory(d, f"IVF{nlist},Flat")
    print("training....")
    index.train(embs)
    index.add(embs)
    faiss.write_index(index, os.path.join(OUT_DIR, "faceworld.index"))
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    print("index built, saved to", OUT_DIR)

if __name__ == "__main__":
    build()
