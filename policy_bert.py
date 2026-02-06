"""
BERT-based policy text encoder for fertility policy embedding.

This module provides:
1. PolicyBertEncoder: Encode policy text using Chinese BERT models
2. KeyArticleExtractor: Extract relevant articles from policy documents
3. PolicyEmbeddingCache: Pre-compute and cache policy embeddings

This is a standalone module that does NOT modify existing code.
To use BERT policy embeddings, use the separate training script: train_multimodal_bert.py

Supported BERT models:
- hfl/chinese-roberta-wwm-ext (recommended)
- hfl/chinese-macbert-base
- bert-base-chinese
- IDEA-CCNL/Erlangshen-Longformer-110M (for very long documents)
"""

import os
import re
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
from tqdm import tqdm

# Lazy imports for transformers (may not be installed)
_transformers_available = None
def _check_transformers():
    global _transformers_available
    if _transformers_available is None:
        try:
            import transformers
            _transformers_available = True
        except ImportError:
            _transformers_available = False
    return _transformers_available


# City to province mapping (imported from policy_features.py for consistency)
from policy_features import CITY_TO_PROVINCE


@dataclass
class PolicyBertConfig:
    """Configuration for BERT policy encoder."""
    model_name: str = "hfl/chinese-roberta-wwm-ext"
    output_dim: int = 64
    max_length: int = 512
    chunk_strategy: str = "hierarchical"  # mean, max, hierarchical, attention
    chunk_overlap: int = 50
    freeze_bert: bool = True
    dropout: float = 0.1
    use_key_articles: bool = True  # Extract only relevant articles
    # Cache settings
    cache_dir: str = "policy_bert_cache"
    policy_lag: int = 1  # Use policy from (year - lag) to prevent data leakage


class KeyArticleExtractor:
    """
    Extract fertility-related articles from policy documents.

    This helps reduce noise and focus on relevant content.
    """

    # Keywords related to fertility policies
    KEYWORDS = [
        # Leave policies
        "产假", "陪产假", "育儿假", "生育假", "护理假", "哺乳假", "婚假",
        # Economic support
        "生育津贴", "育儿补贴", "生育补贴", "生育保险", "生育医疗费",
        # Childcare
        "托育", "托幼", "婴幼儿照护", "托育服务", "普惠托育",
        # Medical
        "辅助生殖", "试管婴儿", "人工授精", "产前检查", "孕产妇",
        # Housing
        "住房公积金", "购房补贴", "住房保障", "多子女家庭住房",
        # General
        "子女", "生育", "三孩", "二孩", "一孩", "多孩",
        "计划生育", "人口", "出生", "婴幼儿",
        # Tax
        "个人所得税", "专项附加扣除", "税收优惠"
    ]

    # Article number patterns
    ARTICLE_PATTERN = re.compile(
        r'(第[一二三四五六七八九十百零\d]+条|第\d+条)'
    )

    def __init__(self, keywords: List[str] = None):
        """
        Args:
            keywords: Custom keywords list. If None, use default.
        """
        self.keywords = keywords or self.KEYWORDS

    def extract_relevant_articles(self, full_text: str) -> str:
        """
        Extract articles containing relevant keywords.

        Args:
            full_text: Complete policy document text

        Returns:
            Concatenated relevant articles
        """
        if not full_text or len(full_text.strip()) == 0:
            return ""

        # Split by article numbers
        parts = self.ARTICLE_PATTERN.split(full_text)

        relevant_parts = []
        current_article_num = ""

        for i, part in enumerate(parts):
            if self.ARTICLE_PATTERN.match(part):
                current_article_num = part
            else:
                full_article = current_article_num + part

                # Check if contains any keyword
                if any(kw in full_article for kw in self.keywords):
                    relevant_parts.append(full_article.strip())

                current_article_num = ""

        # If no articles found, fall back to keyword context extraction
        if not relevant_parts:
            return self.extract_keyword_context(full_text)

        return "\n\n".join(relevant_parts)

    def extract_keyword_context(
        self,
        full_text: str,
        context_window: int = 200
    ) -> str:
        """
        Extract text around keywords when article structure is not clear.

        Args:
            full_text: Complete text
            context_window: Characters before and after keyword

        Returns:
            Concatenated context spans
        """
        spans = []
        seen_positions = set()

        for keyword in self.keywords:
            start = 0
            while True:
                pos = full_text.find(keyword, start)
                if pos == -1:
                    break

                # Avoid overlapping spans
                span_start = max(0, pos - context_window)
                span_end = min(len(full_text), pos + len(keyword) + context_window)

                # Check overlap
                overlap = False
                for seen_start, seen_end in seen_positions:
                    if not (span_end < seen_start or span_start > seen_end):
                        overlap = True
                        break

                if not overlap:
                    spans.append((span_start, span_end, full_text[span_start:span_end]))
                    seen_positions.add((span_start, span_end))

                start = pos + 1

        # Sort by position and concatenate
        spans.sort(key=lambda x: x[0])
        return " ... ".join([s[2] for s in spans])

    def get_relevance_score(self, text: str) -> float:
        """
        Calculate relevance score based on keyword density.

        Args:
            text: Text to score

        Returns:
            Score between 0 and 1
        """
        if not text:
            return 0.0

        keyword_count = sum(text.count(kw) for kw in self.keywords)
        # Normalize by text length (per 100 characters)
        score = min(1.0, keyword_count / (len(text) / 100 + 1))
        return score


class PolicyBertEncoder(nn.Module):
    """
    BERT-based encoder for policy text.

    Supports long documents through chunking strategies:
    - mean: Average pooling of chunk embeddings
    - max: Max pooling of chunk embeddings
    - hierarchical: Transformer aggregation with [CLS] token
    - attention: Learned attention weights over chunks
    """

    def __init__(self, config: PolicyBertConfig = None):
        """
        Args:
            config: PolicyBertConfig instance. If None, use defaults.
        """
        super().__init__()

        if not _check_transformers():
            raise ImportError(
                "transformers library is required for PolicyBertEncoder. "
                "Install with: pip install transformers"
            )

        from transformers import AutoModel, AutoTokenizer

        self.config = config or PolicyBertConfig()

        # Load BERT model and tokenizer
        print(f"Loading BERT model: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.bert = AutoModel.from_pretrained(self.config.model_name)

        # Freeze BERT if specified
        if self.config.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print("BERT parameters frozen")

        hidden_size = self.bert.config.hidden_size  # Usually 768
        self.hidden_size = hidden_size

        # Chunk aggregation components
        if self.config.chunk_strategy == "hierarchical":
            self.chunk_aggregator = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=8,
                    dim_feedforward=hidden_size * 4,
                    dropout=self.config.dropout,
                    batch_first=True
                ),
                num_layers=2
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

        elif self.config.chunk_strategy == "attention":
            self.attention_scorer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.Tanh(),
                nn.Linear(hidden_size // 4, 1)
            )

        # Projection to output dimension
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_size // 2, self.config.output_dim),
            nn.LayerNorm(self.config.output_dim)
        )

        # Key article extractor
        if self.config.use_key_articles:
            self.article_extractor = KeyArticleExtractor()
        else:
            self.article_extractor = None

        self.output_dim = self.config.output_dim

        self._init_weights()

    def _init_weights(self):
        """Initialize non-BERT weights."""
        for module in [self.projection]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Split long text into overlapping chunks.

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        tokens = self.tokenizer.tokenize(text)

        # Reserve space for [CLS] and [SEP]
        chunk_size = self.config.max_length - 2
        overlap = self.config.chunk_overlap

        if len(tokens) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)

            if end >= len(tokens):
                break

            start += chunk_size - overlap

        return chunks

    def _encode_single_chunk(
        self,
        text: str,
        device: torch.device
    ) -> torch.Tensor:
        """
        Encode a single text chunk.

        Args:
            text: Text chunk
            device: Target device

        Returns:
            (1, hidden_size) tensor
        """
        inputs = self.tokenizer(
            text,
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad() if self.config.freeze_bert else torch.enable_grad():
            outputs = self.bert(**inputs)

        # Use [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

    def forward(self, text: str) -> torch.Tensor:
        """
        Encode policy text to embedding.

        Args:
            text: Policy document text (can be long)

        Returns:
            (1, output_dim) policy embedding
        """
        device = next(self.parameters()).device

        # Handle empty text
        if not text or len(text.strip()) == 0:
            return torch.zeros(1, self.output_dim, device=device)

        # Extract relevant articles if enabled
        if self.article_extractor is not None:
            text = self.article_extractor.extract_relevant_articles(text)
            if not text:
                return torch.zeros(1, self.output_dim, device=device)

        # Split into chunks
        chunks = self._split_into_chunks(text)

        # Encode each chunk
        chunk_embeddings = []
        for chunk in chunks:
            emb = self._encode_single_chunk(chunk, device)
            chunk_embeddings.append(emb)

        # Stack: (num_chunks, hidden_size)
        chunk_embeddings = torch.cat(chunk_embeddings, dim=0)
        # Add batch dimension: (1, num_chunks, hidden_size)
        chunk_embeddings = chunk_embeddings.unsqueeze(0)

        # Aggregate chunks
        if self.config.chunk_strategy == "mean":
            pooled = chunk_embeddings.mean(dim=1)

        elif self.config.chunk_strategy == "max":
            pooled = chunk_embeddings.max(dim=1)[0]

        elif self.config.chunk_strategy == "hierarchical":
            # Prepend learnable [CLS] token
            batch_cls = self.cls_token.to(device)
            seq = torch.cat([batch_cls, chunk_embeddings], dim=1)

            # Transformer aggregation
            aggregated = self.chunk_aggregator(seq)
            pooled = aggregated[:, 0, :]  # Take [CLS] position

        elif self.config.chunk_strategy == "attention":
            # Learned attention weights
            scores = self.attention_scorer(chunk_embeddings)  # (1, num_chunks, 1)
            weights = torch.softmax(scores, dim=1)
            pooled = (weights * chunk_embeddings).sum(dim=1)

        else:
            raise ValueError(f"Unknown chunk strategy: {self.config.chunk_strategy}")

        # Project to output dimension
        embedding = self.projection(pooled)
        return embedding

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a batch of texts.

        Args:
            texts: List of policy texts

        Returns:
            (batch_size, output_dim) tensor
        """
        embeddings = []
        for text in texts:
            emb = self.forward(text)
            embeddings.append(emb)
        return torch.cat(embeddings, dim=0)


class MultiDocumentPolicyEncoder(nn.Module):
    """
    Encode multiple policy documents for a province-year and aggregate.

    A province may have multiple relevant documents in a given year:
    - Main regulation (e.g., 人口与计划生育条例)
    - Implementation guidelines
    - Specific support policies
    """

    def __init__(
        self,
        bert_encoder: PolicyBertEncoder,
        aggregation: str = "attention"  # mean, max, attention
    ):
        super().__init__()
        self.bert_encoder = bert_encoder
        self.aggregation = aggregation

        output_dim = bert_encoder.output_dim

        if aggregation == "attention":
            self.doc_attention = nn.Sequential(
                nn.Linear(output_dim, output_dim // 2),
                nn.Tanh(),
                nn.Linear(output_dim // 2, 1)
            )

    def forward(self, documents: List[str]) -> torch.Tensor:
        """
        Encode and aggregate multiple documents.

        Args:
            documents: List of document texts

        Returns:
            (1, output_dim) aggregated embedding
        """
        device = next(self.parameters()).device

        if not documents:
            return torch.zeros(1, self.bert_encoder.output_dim, device=device)

        # Filter empty documents
        documents = [d for d in documents if d and len(d.strip()) > 0]
        if not documents:
            return torch.zeros(1, self.bert_encoder.output_dim, device=device)

        # Encode each document
        doc_embeddings = []
        for doc in documents:
            emb = self.bert_encoder(doc)
            doc_embeddings.append(emb)

        # Stack: (num_docs, output_dim)
        doc_embeddings = torch.cat(doc_embeddings, dim=0)

        if len(documents) == 1:
            return doc_embeddings

        # Aggregate
        if self.aggregation == "mean":
            return doc_embeddings.mean(dim=0, keepdim=True)

        elif self.aggregation == "max":
            return doc_embeddings.max(dim=0, keepdim=True)[0]

        elif self.aggregation == "attention":
            scores = self.doc_attention(doc_embeddings)  # (num_docs, 1)
            weights = torch.softmax(scores, dim=0)
            aggregated = (weights * doc_embeddings).sum(dim=0, keepdim=True)
            return aggregated

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


class PolicyEmbeddingCache:
    """
    Pre-compute and cache policy embeddings for efficient training.

    This avoids running BERT inference during training, which would be slow.
    The cache is built once and loaded at training time.
    """

    def __init__(
        self,
        cache_dir: str = "policy_bert_cache",
        embedding_dim: int = 64,
        policy_lag: int = 1
    ):
        """
        Args:
            cache_dir: Directory to store cache files
            embedding_dim: Dimension of policy embeddings
            policy_lag: Years to lag policy (prevent data leakage)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_dim = embedding_dim
        self.policy_lag = policy_lag
        self.embeddings = {}  # (province, year) -> np.ndarray
        self.default_embedding = np.zeros(embedding_dim, dtype=np.float32)

        # Cache file path
        self.cache_file = self.cache_dir / f"policy_embeddings_dim{embedding_dim}.pkl"

    def build(
        self,
        policy_texts: Dict[Tuple[str, int], Union[str, List[str]]],
        encoder: Union[PolicyBertEncoder, MultiDocumentPolicyEncoder],
        device: str = "cuda",
        batch_size: int = 1
    ):
        """
        Build the embedding cache.

        Args:
            policy_texts: Dict mapping (province, year) to text or list of texts
            encoder: BERT encoder instance
            device: Device to run inference on
            batch_size: Batch size for encoding (not used currently)
        """
        encoder = encoder.to(device)
        encoder.eval()

        print(f"Building policy embedding cache...")
        print(f"  Total entries: {len(policy_texts)}")
        print(f"  Output dimension: {self.embedding_dim}")
        print(f"  Cache file: {self.cache_file}")

        with torch.no_grad():
            for (province, year), text_data in tqdm(
                policy_texts.items(),
                desc="Encoding policies"
            ):
                # Handle single text or list of texts
                if isinstance(text_data, list):
                    if isinstance(encoder, MultiDocumentPolicyEncoder):
                        embedding = encoder(text_data)
                    else:
                        # Concatenate texts for single-doc encoder
                        combined_text = "\n\n".join(text_data)
                        embedding = encoder(combined_text)
                else:
                    embedding = encoder(text_data)

                # Store as numpy
                self.embeddings[(province, year)] = embedding.squeeze().cpu().numpy()

        self.save()
        print(f"Cache built with {len(self.embeddings)} entries")

    def get(self, province: str, year: int, apply_lag: bool = True) -> np.ndarray:
        """
        Get policy embedding for a province-year.

        Args:
            province: Province name
            year: Target year (for prediction)
            apply_lag: Whether to apply policy_lag

        Returns:
            Policy embedding as numpy array
        """
        # Apply temporal lag to prevent data leakage
        policy_year = year - self.policy_lag if apply_lag else year

        key = (province, policy_year)
        if key in self.embeddings:
            return self.embeddings[key]

        # Fallback: try earlier years
        for fallback_year in range(policy_year, 2012, -1):
            fallback_key = (province, fallback_year)
            if fallback_key in self.embeddings:
                return self.embeddings[fallback_key]

        # Ultimate fallback: zero embedding
        return self.default_embedding.copy()

    def get_for_city(self, city: str, year: int, apply_lag: bool = True) -> np.ndarray:
        """
        Get policy embedding for a city-year (maps city to province).

        Args:
            city: City name
            year: Target year
            apply_lag: Whether to apply policy_lag

        Returns:
            Policy embedding
        """
        province = CITY_TO_PROVINCE.get(city, city)
        return self.get(province, year, apply_lag=apply_lag)

    def save(self):
        """Save cache to file."""
        cache_data = {
            'embeddings': self.embeddings,
            'embedding_dim': self.embedding_dim,
            'policy_lag': self.policy_lag
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Cache saved to {self.cache_file}")

    def load(self, expected_dim: int = None, strict: bool = True) -> bool:
        """
        Load cache from file.

        Args:
            expected_dim: Expected embedding dimension. If provided, validates against
                         the cache's actual dimension.
            strict: If True (default), raise error on dimension mismatch.
                   If False, print warning but continue.

        Returns:
            True if loaded successfully, False otherwise

        Raises:
            ValueError: If strict=True and expected_dim doesn't match cache dimension.
        """
        if not self.cache_file.exists():
            print(f"Cache file not found: {self.cache_file}")
            return False

        with open(self.cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        cached_dim = cache_data['embedding_dim']

        # Validate dimension if expected_dim is provided
        if expected_dim is not None and cached_dim != expected_dim:
            error_msg = (
                f"BERT cache dimension mismatch!\n"
                f"  Cache file: {self.cache_file}\n"
                f"  Cache dimension: {cached_dim}\n"
                f"  Expected dimension: {expected_dim}\n"
                f"  Solution: Rebuild the cache with matching dimension:\n"
                f"    python build_policy_bert_cache.py --dim {expected_dim}"
            )
            if strict:
                raise ValueError(error_msg)
            else:
                print(f"WARNING: {error_msg}")
                print("Continuing with cache dimension (may cause runtime errors)...")

        self.embeddings = cache_data['embeddings']
        self.embedding_dim = cached_dim
        self.policy_lag = cache_data.get('policy_lag', 1)

        print(f"Cache loaded: {len(self.embeddings)} entries, dim={self.embedding_dim}")
        return True

    def get_statistics(self) -> dict:
        """Get cache statistics."""
        if not self.embeddings:
            return {'count': 0}

        embeddings_array = np.stack(list(self.embeddings.values()))

        return {
            'count': len(self.embeddings),
            'embedding_dim': self.embedding_dim,
            'policy_lag': self.policy_lag,
            'mean_norm': float(np.linalg.norm(embeddings_array, axis=1).mean()),
            'provinces': len(set(k[0] for k in self.embeddings.keys())),
            'year_range': (
                min(k[1] for k in self.embeddings.keys()),
                max(k[1] for k in self.embeddings.keys())
            )
        }


def load_policy_texts_from_json(
    json_path: str,
    text_field: str = "full_text",
    fallback_fields: List[str] = None
) -> Dict[Tuple[str, int], List[str]]:
    """
    Load policy texts from JSON file.

    Expected JSON structure:
    {
        "省份名": {
            "2021": {
                "documents": [
                    {"title": "...", "full_text": "..."},
                    ...
                ]
            }
        }
    }

    Or flat structure:
    {
        "policies": [
            {"province": "北京市", "year": 2021, "full_text": "..."}
        ]
    }

    Args:
        json_path: Path to JSON file
        text_field: Field name containing full text
        fallback_fields: Fallback fields if text_field is missing

    Returns:
        Dict mapping (province, year) to list of document texts
    """
    fallback_fields = fallback_fields or ["main_content", "content", "text"]

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = {}

    # Try nested structure first
    if isinstance(data, dict) and not data.get("policies"):
        for province, years_data in data.items():
            if province in ["metadata", "national_policies"]:
                continue

            if not isinstance(years_data, dict):
                continue

            for year_str, year_data in years_data.items():
                try:
                    year = int(year_str)
                except ValueError:
                    continue

                texts = []

                # Get documents
                docs = year_data.get("documents", [year_data])
                if not isinstance(docs, list):
                    docs = [docs]

                for doc in docs:
                    if not isinstance(doc, dict):
                        continue

                    # Try text_field first, then fallbacks
                    text = doc.get(text_field)
                    if not text:
                        for fallback in fallback_fields:
                            text = doc.get(fallback)
                            if text:
                                break

                    if text and len(text.strip()) > 0:
                        texts.append(text)

                if texts:
                    result[(province, year)] = texts

    # Try flat structure
    elif "policies" in data:
        for policy in data["policies"]:
            province = policy.get("province")
            year = policy.get("year")

            if not province or not year:
                continue

            text = policy.get(text_field)
            if not text:
                for fallback in fallback_fields:
                    text = policy.get(fallback)
                    if text:
                        break

            if text and len(text.strip()) > 0:
                key = (province, int(year))
                if key not in result:
                    result[key] = []
                result[key].append(text)

    return result


def get_policy_bert_cache(
    cache_dir: str = None,
    embedding_dim: int = 64,
    policy_lag: int = 1,
    validate_dim: bool = True,
    strict: bool = True
) -> Optional[PolicyEmbeddingCache]:
    """
    Get or create policy embedding cache.

    Args:
        cache_dir: Cache directory. If None, use default.
        embedding_dim: Expected embedding dimension. Used for validation.
        policy_lag: Policy temporal lag
        validate_dim: Whether to validate cache dimension against embedding_dim.
                     Default True for safety.
        strict: If True, raise error on dimension mismatch.
               If False, print warning but continue.

    Returns:
        PolicyEmbeddingCache if cache exists, None otherwise

    Raises:
        ValueError: If validate_dim=True, strict=True, and cache dimension
                   doesn't match embedding_dim.
    """
    if cache_dir is None:
        cache_dir = Path(__file__).parent / "policy_bert_cache"

    cache = PolicyEmbeddingCache(
        cache_dir=str(cache_dir),
        embedding_dim=embedding_dim,
        policy_lag=policy_lag
    )

    expected_dim = embedding_dim if validate_dim else None
    if cache.load(expected_dim=expected_dim, strict=strict):
        return cache

    return None


if __name__ == "__main__":
    # Test the module
    print("=" * 60)
    print("Testing PolicyBertEncoder")
    print("=" * 60)

    # Test KeyArticleExtractor
    print("\n1. Testing KeyArticleExtractor")
    extractor = KeyArticleExtractor()

    test_text = """
    第一条 为了保护公民的合法权益，制定本条例。
    第十六条 依法办理结婚登记的夫妻，女方除享受国家规定的产假外，
    享受延长生育假六十日，男方享受陪产假十五日。
    第二十条 市、区人民政府应当建立育儿补贴制度。
    第五十条 本条例自公布之日起施行。
    """

    relevant = extractor.extract_relevant_articles(test_text)
    print(f"Original length: {len(test_text)}")
    print(f"Extracted length: {len(relevant)}")
    print(f"Relevance score: {extractor.get_relevance_score(test_text):.2f}")

    # Test PolicyBertEncoder (if transformers available)
    print("\n2. Testing PolicyBertEncoder")
    if _check_transformers():
        config = PolicyBertConfig(
            model_name="hfl/chinese-roberta-wwm-ext",
            output_dim=64,
            chunk_strategy="mean",
            freeze_bert=True
        )

        print(f"Config: {config}")

        # Only run if GPU available (BERT is slow on CPU)
        if torch.cuda.is_available():
            encoder = PolicyBertEncoder(config)
            encoder = encoder.cuda()

            embedding = encoder(test_text)
            print(f"Embedding shape: {embedding.shape}")
            print(f"Embedding norm: {embedding.norm().item():.4f}")
        else:
            print("Skipping BERT test (no GPU available)")
    else:
        print("transformers not installed, skipping BERT test")

    # Test PolicyEmbeddingCache
    print("\n3. Testing PolicyEmbeddingCache")
    cache = PolicyEmbeddingCache(
        cache_dir="/tmp/test_policy_cache",
        embedding_dim=64
    )

    # Simulate cached embeddings
    cache.embeddings[("北京市", 2021)] = np.random.randn(64).astype(np.float32)
    cache.embeddings[("上海市", 2021)] = np.random.randn(64).astype(np.float32)

    # Test retrieval
    emb = cache.get_for_city("北京市", 2022, apply_lag=True)
    print(f"Retrieved embedding shape: {emb.shape}")

    emb_unknown = cache.get_for_city("未知城市", 2022)
    print(f"Unknown city embedding (zeros): {np.allclose(emb_unknown, 0)}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
