"""SamGeo3 adapter. Imported only by the isolated model worker."""
from __future__ import annotations

import os
import re


class Sam3:
    engine = "sam3"

    def __init__(self, device=None, token=None, report=None):
        import torch
        from huggingface_hub import get_token
        from samgeo.samgeo3 import SamGeo3
        from importlib.metadata import version

        # SamGeo3 does not forward a token to from_pretrained. Set it only in this worker;
        # never call login(), write the token, or pass it on the process command line.
        token = token or get_token()
        if not token:
            raise RuntimeError("SAM3_ACCESS_REQUIRED: Request Meta approval at https://huggingface.co/facebook/sam3, "
                               "then supply your own Hugging Face read token (session field, HF_TOKEN or hf auth login).")
        os.environ["HF_TOKEN"] = token
        if device is None:
            mps = getattr(torch.backends, "mps", None)
            device = "cuda" if torch.cuda.is_available() else "mps" if mps and mps.is_available() else "cpu"
        if device not in ("cuda", "mps", "cpu"):
            raise ValueError("device must be cuda, mps or cpu")
        self.device = device
        if report is not None:
            try:
                from .worker import download
            except ImportError:
                from worker import download
            download([("facebook/sam3", ["*.json", "*.txt", "*.safetensors", "*.model"])], report, token)
        self.sam = SamGeo3(backend="transformers", model_id="facebook/sam3", device=device)
        self.sam.model.eval()
        self.versions = {name: str(version(dist)) for name, dist in
                         (("torch", "torch"), ("transformers", "transformers"), ("samgeo", "segment-geospatial"))}

    def segment(self, image, prompt, box_threshold, text_threshold):
        import numpy as np
        try:
            from .worker import _cut
        except ImportError:
            from worker import _cut

        if not 0 <= box_threshold <= 1:
            raise ValueError("confidence threshold must be between 0 and 1")
        self.sam.set_image(image)
        self.sam.set_confidence_threshold(box_threshold)
        objects = []
        # DINO takes period-separated concepts; SAM 3 expects one concept per request.
        # Preserve multiword phrases and retain their labels in the output.
        concepts = list(dict.fromkeys(p.strip() for p in re.split(r"[.;,]+", prompt) if p.strip()))
        for label in concepts:
            self.sam.generate_masks(label, quiet=True)
            for raw, score in zip(self.sam.masks, self.sam.scores):
                mask = np.asarray(raw, dtype=bool).reshape(image.shape[:2])
                score = float(score)
                if not np.isfinite(score) or score < box_threshold:
                    continue
                cut = _cut(mask)
                if cut is None:
                    continue
                r, c, h, w, _ = cut
                # No mask-quality estimate is exposed by this backend. Do not invent one.
                objects.append({"box": [float(c), float(r), float(c + w), float(r + h)],
                                "score": score, "quality": None, "label": label, "mask": cut})
        return objects
