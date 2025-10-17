# backend/app/utils/faiss_indexer.py
import faiss, numpy as np, json, os

class FaissIndex:
    def __init__(self, index_dir):
        self.index_dir = index_dir
        self.index = None
        self.metadata = {}
        # expected files: index_file, metadata.json, embeddings.npy
        self.index_file = os.path.join(index_dir, "faceworld.index")
        self.meta_file = os.path.join(index_dir, "metadata.json")
        self.load()

    def load(self):
        if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.meta_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            # empty index placeholder (512 dims)
            self.index = faiss.IndexFlatL2(512)
            self.metadata = {}

    def search(self, query_vector, top_k=3, domain="celebrity"):
        # query_vector: numpy array shape (512,)
        if query_vector.ndim==1:
            q = query_vector.reshape(1,-1).astype('float32')
        else:
            q = query_vector.astype('float32')
        D, I = self.index.search(q, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            meta = self.metadata.get(str(int(idx)), {})
            # you can filter by domain key in metadata
            if domain and meta.get("domain") != domain and domain!="all":
                continue
            results.append({"id": int(idx), "score": float(score), "path": meta.get("path","")})
        return results
