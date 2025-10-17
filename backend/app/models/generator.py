# backend/app/models/generator.py
from PIL import Image, ImageFilter, ImageOps
import os

class Generator:
    def __init__(self, model_dir=None):
        self.model_dir = model_dir

    def generate(self, pil_img, style="cartoon"):
        # placeholder fast CPU-friendly stylizers for demo
        if style == "cartoon":
            out = pil_img.filter(ImageFilter.EDGE_ENHANCE_MORE)
            out = ImageOps.posterize(out, 3)
            return out
        elif style == "anime":
            out = pil_img.filter(ImageFilter.SHARPEN)
            return out
        elif style == "celebrity":
            # fake stylize: slight smoothing + color shift
            out = pil_img.filter(ImageFilter.SMOOTH_MORE).convert("RGB")
            return out
        else:
            return pil_img
