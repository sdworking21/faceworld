# backend/app/utils/preprocess.py
from PIL import Image
import numpy as np
import cv2
# We will use RetinaFace if available. Fallback to OpenCV frontal face if not.

try:
    from retinaface import RetinaFace
    HAVE_RETINA = True
except Exception:
    HAVE_RETINA = False
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def align_and_crop(pil_img, target_size=(256,256)):
    img_cv = pil_to_cv(pil_img)
    h, w = img_cv.shape[:2]
    if HAVE_RETINA:
        try:
            resp = RetinaFace.detect_faces(img_cv)
            # get the largest face
            if isinstance(resp, dict):
                # pick largest
                face_key = max(resp.items(), key=lambda x: (x[1]['facial_area'][2]-x[1]['facial_area'][0]))[0]
                bbox = resp[face_key]["facial_area"]  # x1,y1,x2,y2
                x1,y1,x2,y2 = bbox
            else:
                # no faces
                return None
        except Exception:
            return None
    else:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces)==0:
            return None
        x,y,wf,hf = max(faces, key=lambda x: x[2]*x[3])
        x1,y1,x2,y2 = x,y,x+wf,y+hf
    # crop + resize center
    margin = int(0.25 * max(x2-x1, y2-y1))
    x1m = max(0, x1 - margin); y1m = max(0, y1 - margin)
    x2m = min(w, x2 + margin); y2m = min(h, y2 + margin)
    crop = img_cv[y1m:y2m, x1m:x2m]
    resized = cv2.resize(crop, target_size)
    return cv_to_pil(resized)
