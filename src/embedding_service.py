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

    def build_job_text(self, job):
        """
        Ghep cac truong quan trong cua job thanh 1 doan text de embed.
        Uu tien: title > requirements_tags > technical_skills > job_requirements

        Args:
            job: dict chua thong tin job

        Returns:
            str - text da ghep
        """
        parts = []

        title = str(job.get("title", "")).strip()
        if title:
            parts.append(title)

        req_tags = str(job.get("requirements_tags", "")).strip()
        if req_tags:
            parts.append(req_tags)

        tech_skills = str(job.get("technical_skills", "")).strip()
        if tech_skills:
            parts.append(tech_skills)

        specializations = str(job.get("specializations", "")).strip()
        if specializations:
            parts.append(specializations)

        job_req = str(job.get("job_requirements", "")).strip()
        if job_req:
            parts.append(job_req[:500])

        return ". ".join(parts) if parts else ""

    def build_cv_text(self, cv_data):
        """
        Ghep thong tin CV thanh 1 doan text de embed.

        Args:
            cv_data: dict chua thong tin CV

        Returns:
            str - text da ghep
        """
        parts = []

        skills = str(cv_data.get("skills", "")).strip()
        if skills:
            parts.append(skills)

        profile = str(cv_data.get("profile_text", "")).strip()
        if profile:
            parts.append(profile)

        return ". ".join(parts) if parts else skills


def get_embedding_service(device=None):
    """
    Singleton pattern - chi tao model 1 lan.
    """
    global _instance
    if _instance is None:
        _instance = EmbeddingService(device=device)
    return _instance


if __name__ == "__main__":
    service = get_embedding_service()

    test_texts = [
        "Python, React, Node.js, fullstack developer",
        "Lap trinh vien Python 3 nam kinh nghiem",
        "Project Management, quan ly du an",
        "Ke toan truong, phan tich tai chinh",
    ]

    print(f"\nTest embedding {len(test_texts)} texts:")
    embeddings = service.encode(test_texts)
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")

    from numpy import dot
    from numpy.linalg import norm

    print("\nCosine similarity matrix:")
    for i in range(len(test_texts)):
        sims = []
        for j in range(len(test_texts)):
            sim = dot(embeddings[i], embeddings[j])
            sims.append(f"{sim:.3f}")
        print(f"  [{i}] {' | '.join(sims)}  <- {test_texts[i][:40]}")
