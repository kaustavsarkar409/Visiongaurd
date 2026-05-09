from __future__ import annotations

from dataclasses import dataclass
import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


@dataclass
class ActionPrediction:
    label: str
    confidence: float
    violence_score: float
    is_violence: bool


class ViolenceActionRecognizer:
    def __init__(
        self,
        confidence_threshold: float = 0.35,
        device: Optional[str] = None,
        cache_dir: str = ".torch-cache",
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.device = device or self._pick_device()
        self.cache_dir = cache_dir
        self.model = None
        self.categories: Sequence[str] = []
        self.enabled = False
        self.last_error: Optional[str] = None
        self.violence_keywords = (
            "punch",
            "kick",
            "fight",
            "hit",
            "wrestling",
            "sword",
            "shoot",
            "gun",
            "slap",
            "karate",
            "boxing",
            "martial",
        )
        self._load_model()

    def _pick_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self) -> None:
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.hub.set_dir(self.cache_dir)
            from torchvision.models.video import R3D_18_Weights, r3d_18

            weights = R3D_18_Weights.DEFAULT
            model = r3d_18(weights=weights)
            model.eval()
            model.to(self.device)
            self.model = model
            self.categories = weights.meta.get("categories", [])
            self.enabled = True
        except Exception as exc:
            self.enabled = False
            self.last_error = str(exc)

    def _preprocess_frames(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        resized = [
            cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (112, 112))
            for frame in frames_bgr
        ]
        clip = np.stack(resized, axis=0).astype(np.float32) / 255.0

        mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
        std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
        clip = (clip - mean) / std

        clip = np.transpose(clip, (3, 0, 1, 2))
        tensor = torch.from_numpy(clip).unsqueeze(0).to(self.device)
        return tensor

    def _violence_score_from_topk(
        self, topk: List[Tuple[str, float]]
    ) -> Tuple[float, Optional[Tuple[str, float]]]:
        scores = []
        best_label = None
        for label, score in topk:
            label_l = label.lower()
            if any(keyword in label_l for keyword in self.violence_keywords):
                scores.append(score)
                if best_label is None or score > best_label[1]:
                    best_label = (label, score)
        violence_score = float(max(scores)) if scores else 0.0
        return violence_score, best_label

    def predict(self, frames_bgr: List[np.ndarray]) -> ActionPrediction:
        if not self.enabled or self.model is None:
            return ActionPrediction(
                label="Action model unavailable",
                confidence=0.0,
                violence_score=0.0,
                is_violence=False,
            )

        with torch.no_grad():
            inputs = self._preprocess_frames(frames_bgr)
            logits = self.model(inputs)
            probs = torch.softmax(logits[0], dim=0)
            topk_vals, topk_idx = torch.topk(probs, k=5)

        topk = []
        for idx, val in zip(topk_idx.tolist(), topk_vals.tolist()):
            label = self.categories[idx] if idx < len(self.categories) else f"class_{idx}"
            topk.append((label, float(val)))

        best_label, best_conf = topk[0]
        violence_score, matched = self._violence_score_from_topk(topk)
        chosen_label = matched[0] if matched else best_label
        chosen_conf = matched[1] if matched else best_conf
        is_violence = violence_score >= self.confidence_threshold

        return ActionPrediction(
            label=chosen_label,
            confidence=float(chosen_conf),
            violence_score=violence_score,
            is_violence=is_violence,
        )
