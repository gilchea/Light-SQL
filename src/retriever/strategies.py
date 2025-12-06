import faiss
import random
import logging
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils import mask_question

logger = logging.getLogger(__name__)

class BaseRetriever:
    """Abstract base class for retrieval strategies."""
    def build_index(self, examples: list):
        pass
    def retrieve(self, query: str, k: int) -> list:
        return []

class RandomRetriever(BaseRetriever):
    """Retrieves k random examples."""
    def __init__(self):
        self.examples = []

    def build_index(self, examples: list):
        self.examples = examples

    def retrieve(self, query: str, k: int) -> list:
        if not self.examples or k == 0: return []
        return random.sample(self.examples, min(k, len(self.examples)))

class AdvancedRetriever(BaseRetriever):
    """
    Retrieves examples using Semantic Search (Dense Retrieval).
    Supports multiple modes:
    - qts: Question-to-SQL (Standard)
    - mqs: Masked-Question-to-SQL (Structure based)
    - qrs: Query-to-SQL (Retrieves based on generated SQL draft)
    """
    def __init__(self, model_name: str, mode: str = 'qts', device: str = 'cuda'):
        self.model = SentenceTransformer(model_name, device=device)
        self.index = None
        self.examples = []
        self.mode = mode
        self.device = device if torch.cuda.is_available() else 'cpu'

    def _get_text_to_embed(self, item: dict) -> str:
        """Determines what text to embed based on the mode."""
        if self.mode == 'qts': return item['question']
        elif self.mode == 'mqs': return mask_question(item['question'])
        elif self.mode == 'qrs': return item['query'] # Embeds the SQL query
        return item['question']

    def build_index(self, examples: list):
        """Encodes examples and builds a FAISS index."""
        self.examples = examples
        if not examples: return

        # Encode without progress bar for cleaner output
        texts = [self._get_text_to_embed(ex) for ex in examples]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        if self.device == 'cuda':
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception: 
                pass # Fallback to CPU if GPU memory is full/unavailable

        self.index.add(embeddings)

    def retrieve(self, query: str, k: int) -> list:
        if not self.index or k == 0: return []

        search_text = query
        if self.mode == 'mqs':
            search_text = mask_question(query)

        q_emb = self.model.encode([search_text], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(q_emb)
        
        # Search in FAISS index
        _, I = self.index.search(q_emb, k)
        
        # Return examples, ignoring invalid indices (-1)
        return [self.examples[i] for i in I[0] if i != -1]