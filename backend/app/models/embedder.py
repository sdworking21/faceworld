# backend/app/models/embedder.py
import numpy as np
try:
    import insightface
    HAVE_INSIGHT = True
except Exception:
    HAVE_INSIGHT = False

def load_embedder():
    if HAVE_INSIGHT:
        model = insightface.app.FaceAnalysis(allowed_modules=['detection','recognition'])
        model.prepare(ctx_id=0, det_size=(640,640))
        return model
    else:
        raise RuntimeError("InsightFace not installed; please pip install insightface")

def get_embedding(model, pil_image):
    # pil_image: PIL image RGB
    img = np.asarray(pil_image)[:,:,::-1]  # RGB->BGR for insightface
    faces = model.get(img)
    if not faces:
        raise RuntimeError("No face found by embedder")
    return np.array(faces[0].embedding).astype('float32')
