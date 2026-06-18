"""
Embedding Service - bge-m3 model cho Semantic Search
Cung cap API encode text thanh vector 1024 chieu
"""

import os
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_instance = None


class EmbeddingService:
    MODEL_NAME = "BAAI/bge-m3"
    DIMS = 1024

    def __init__(self, device=None):
        """
        Khoi tao bge-m3 model.
        Tu dong chon GPU neu co, fallback CPU.

        Args:
            device: 'cuda', 'cpu', hoac None (tu dong)
        """
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        logger.info(f"Dang tai model {self.MODEL_NAME} tren {device}...")
        self.model = SentenceTransformer(self.MODEL_NAME, device=device)
        self.device = device
        logger.info(f"Model da san sang ({device}). Output dims = {self.DIMS}")

    def encode(self, texts, batch_size=32, show_progress=False):
        """
        Encode danh sach text thanh vectors.

        Args:
            texts: list[str] hoac str
            batch_size: kich thuoc batch
            show_progress: hien thanh tien do

        Returns:
            numpy array shape (n, 1024)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return embeddings

    def encode_single(self, text):
        """
        Encode 1 doan text thanh vector.

        Args:
            text: str

        Returns:
            list[float] - vector 1024 chieu
        """
        if not text or not text.strip():
            return [0.0] * self.DIMS

        embedding = self.model.encode(
            text.strip(),
            normalize_embeddings=True,
        )
        return embedding.tolist()

    def build_cv_text(self, cv_data):
        """
        Ghep thong tin CV thanh 1 doan text de embed.

        Args:
            cv_data: dict chua thong tin CV

        Returns:
            str - text da ghep
        """
        skills = str(cv_data.get("skills", "")).strip()
        return skills


def get_embedding_service(device=None):
    """
    Singleton pattern - chi tao model 1 lan.
    """
    global _instance
    if _instance is None:
        _instance = EmbeddingService(device=device)
    return _instance

