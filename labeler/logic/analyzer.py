import os
import re
import json
import hashlib
import logging
import pickle
import csv
import gc
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from contextlib import asynccontextmanager
import time

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import openai
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP client logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration with memory-conscious defaults."""
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
    LLM_MODEL: str = os.getenv('LLM_MODEL', 'gpt-4o-mini')
    
    # Rate limit safety: process in small batches with delays
    EMBEDDING_BATCH_SIZE: int = 50  # Conservative batch size
    LLM_BATCH_SIZE: int = 10  # Claims to extract per LLM call
    LLM_DELAY_SECONDS: float = 0.5  # Delay between LLM calls
    
    # Clustering
    CLUSTERING_EPS: float = float(os.getenv('CLUSTERING_EPS', '0.25'))  # Tighter clusters
    CLUSTERING_MIN_SAMPLES: int = int(os.getenv('CLUSTERING_MIN_SAMPLES', '2'))
    
    # Detection thresholds - HIGHER now that we're matching claims
    SIMILARITY_THRESHOLD: float = float(os.getenv('SIMILARITY_THRESHOLD', '0.60'))  # Raised from 0.45
    HIGH_CONFIDENCE_THRESHOLD: float = float(os.getenv('HIGH_CONFIDENCE_THRESHOLD', '0.75'))
    
    # Data paths
    IS_PRODUCTION: bool = os.getenv('NODE_ENV') == 'production'
    DATA_DIR: str = '/data' if IS_PRODUCTION else './data'
    CACHE_DIR: str = f"{DATA_DIR}/cache"
    NOTES_FILE: str = os.getenv('NOTES_FILE', f"{DATA_DIR}/notes-00000.tsv")
    CLAIMS_CACHE_FILE: str = f"{DATA_DIR}/cache/extracted_claims.json"
    LABELS_OUTPUT_FILE: str = f"{DATA_DIR}/generated-labels.json"
    
    # Limits - bounded for memory safety
    MAX_NOTES: int = int(os.getenv('MAX_NOTES', '500'))
    MAX_CLUSTERS: int = int(os.getenv('MAX_CLUSTERS', '30'))
    MAX_KEYWORDS_PER_CLUSTER: int = 15
    MAX_CLAIMS_PER_CLUSTER: int = 5  # Sample claims to keep for context
    
    @classmethod
    def validate(cls) -> List[str]:
        errors = []
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
        return errors

CONFIG = Config()

# ============================================================================
# DATA CLASSES (Memory-Bounded)
# ============================================================================

@dataclass
class ExtractedClaim:
    """A false claim extracted from a Community Note."""
    note_id: str
    original_summary: str  # Keep truncated for debugging
    false_claim: str  # The extracted misinformation claim
    claim_keywords: List[str]  # Specific keywords from this claim
    embedding: Optional[np.ndarray] = None


@dataclass
class ClaimCluster:
    """
    A cluster of similar false claims.
    Memory-bounded: only stores centroid + metadata, not all claims.
    """
    cluster_id: int
    centroid: np.ndarray  # float32, ~6KB
    label_id: str
    label_name: str
    label_description: str
    keywords: Set[str]  # Union of claim keywords, bounded by MAX_KEYWORDS_PER_CLUSTER
    sample_claims: List[str]  # 3-5 example claims for logging/debugging
    severity: str = "inform"
    claim_count: int = 0


@dataclass
class LabelDefinition:
    """Bluesky label definition."""
    identifier: str
    severity: str
    blurs: str
    defaultSetting: str
    locales: List[Dict[str, str]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "severity": self.severity,
            "blurs": self.blurs,
            "defaultSetting": self.defaultSetting,
            "locales": self.locales
        }


# ============================================================================
# CLAIM EXTRACTOR
# ============================================================================

class ClaimExtractor:
    """
    Extracts false claims from Community Note summaries using LLM.
    Results are cached to disk to avoid re-extraction on restart.
    """
    
    def __init__(self, client: openai.OpenAI):
        self.client = client
        self.cache_path = Path(CONFIG.CLAIMS_CACHE_FILE)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_cache(self) -> Dict[str, dict]:
        """Load cached claim extractions from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load claims cache: {e}")
        return {}
    
    def _save_cache(self, cache: Dict[str, dict]):
        """Save claim extractions to disk."""
        try:
            with open(self.cache_path, 'w') as f:
                json.dump(cache, f)
        except Exception as e:
            logger.error(f"Failed to save claims cache: {e}")
    
    def extract_claims_batch(self, summaries: List[Tuple[str, str]]) -> List[Optional[dict]]:
        """
        Extract false claims from a batch of summaries.
        
        Args:
            summaries: List of (note_id, summary_text) tuples
            
        Returns:
            List of {false_claim, keywords} dicts or None for failures
        """
        # Load existing cache
        cache = self._load_cache()
        results: List[Optional[dict]] = []
        uncached: List[Tuple[int, str, str, str]] = []  # (index, note_id, summary, cache_key)
        
        # Check cache first
        for i, (note_id, summary) in enumerate(summaries):
            cache_key = hashlib.sha256(summary.encode()).hexdigest()[:16]
            if cache_key in cache:
                results.append(cache[cache_key])
            else:
                results.append(None)  # Placeholder
                uncached.append((i, note_id, summary, cache_key))
        
        if not uncached:
            logger.info(f"All {len(summaries)} claims loaded from cache")
            return results
        
        logger.info(f"Extracting {len(uncached)} claims (cached: {len(summaries) - len(uncached)})")
        
        # Process uncached in batches to respect rate limits
        batch_size = CONFIG.LLM_BATCH_SIZE
        new_cache_entries = {}
        
        for batch_start in range(0, len(uncached), batch_size):
            batch = uncached[batch_start:batch_start + batch_size]
            
            # Build prompt for batch extraction
            prompt_parts = ["Extract the FALSE CLAIM being corrected from each fact-check summary below.\n"]
            prompt_parts.append("For each summary, identify:\n")
            prompt_parts.append("1. The specific false claim (what misinformation is being spread)\n")
            prompt_parts.append("2. 3-5 specific keywords that would appear in posts making this claim\n\n")
            
            for j, (idx, note_id, summary, cache_key) in enumerate(batch):
                # Truncate long summaries
                truncated = summary[:500] if len(summary) > 500 else summary
                prompt_parts.append(f"[{j}] {truncated}\n\n")
            
            prompt_parts.append("Respond with JSON array. For each summary, provide:\n")
            prompt_parts.append('{"index": N, "false_claim": "the claim being spread", "keywords": ["specific", "terms"]}\n')
            prompt_parts.append("If no clear false claim exists, use null for that entry.\n")
            prompt_parts.append("Keywords should be SPECIFIC to the misinformation, not generic topic words.")
            
            prompt = "".join(prompt_parts)
            
            try:
                response = self.client.chat.completions.create(
                    model=CONFIG.LLM_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You extract false claims from fact-check summaries. Be precise - extract the MISINFORMATION, not the correction. Output valid JSON only."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                
                # Parse response
                content = response.choices[0].message.content.strip()
                # Handle markdown code blocks
                if content.startswith("```"):
                    content = re.sub(r'^```\w*\n?', '', content)
                    content = re.sub(r'\n?```$', '', content)
                
                extracted = json.loads(content)
                
                # Map results back
                for item in extracted:
                    if item is None:
                        continue
                    j = item.get("index")
                    if j is not None and 0 <= j < len(batch):
                        idx, note_id, summary, cache_key = batch[j]
                        result = {
                            "false_claim": item.get("false_claim", ""),
                            "keywords": item.get("keywords", [])[:10]  # Bound keywords
                        }
                        if result["false_claim"]:
                            results[idx] = result
                            new_cache_entries[cache_key] = result
                
                logger.info(f"Extracted claims batch {batch_start//batch_size + 1}/{(len(uncached) + batch_size - 1)//batch_size}")
                
                # Save cache AFTER EACH BATCH to survive restarts
                if new_cache_entries:
                    cache.update(new_cache_entries)
                    self._save_cache(cache)
                
            except Exception as e:
                logger.error(f"Claim extraction failed for batch: {e}")
            
            # Rate limit delay
            if batch_start + batch_size < len(uncached):
                time.sleep(CONFIG.LLM_DELAY_SECONDS)
        
        return results


# ============================================================================
# EMBEDDING SERVICE (Memory-Optimized)
# ============================================================================

class EmbeddingService:
    """
    Handles embeddings with disk caching only.
    No in-memory cache to prevent unbounded growth.
    """
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = CONFIG.EMBEDDING_MODEL
        self.cache_path = Path(CONFIG.CACHE_DIR) / 'embeddings.pkl'
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._disk_cache: Dict[str, np.ndarray] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load embeddings cache from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'rb') as f:
                    self._disk_cache = pickle.load(f)
                logger.info(f"Loaded {len(self._disk_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Could not load embedding cache: {e}")
                self._disk_cache = {}
    
    def _save_cache(self):
        """Save embeddings to disk."""
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self._disk_cache, f)
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")
    
    def _clear_memory(self):
        """Clear in-memory cache after initialization."""
        self._disk_cache = {}
        gc.collect()
    
    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:24]
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get single embedding (for runtime post analysis)."""
        try:
            response = self.client.embeddings.create(
                input=text[:8000],  # Truncate to model limit
                model=self.model
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None
    
    def get_embeddings_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Get embeddings for initialization (uses disk cache)."""
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        uncached: List[Tuple[int, str]] = []
        
        # Check cache
        for i, text in enumerate(texts):
            h = self._hash(text)
            if h in self._disk_cache:
                results[i] = self._disk_cache[h]
            else:
                uncached.append((i, text))
        
        if not uncached:
            return results
        
        # Batch API calls
        batch_size = CONFIG.EMBEDDING_BATCH_SIZE
        for batch_start in range(0, len(uncached), batch_size):
            batch = uncached[batch_start:batch_start + batch_size]
            texts_batch = [t[:8000] for _, t in batch]
            
            try:
                response = self.client.embeddings.create(
                    input=texts_batch,
                    model=self.model
                )
                
                for j, emb_data in enumerate(response.data):
                    idx, text = batch[j]
                    embedding = np.array(emb_data.embedding, dtype=np.float32)
                    results[idx] = embedding
                    self._disk_cache[self._hash(text)] = embedding
                
                logger.info(f"Embedded batch {batch_start//batch_size + 1}/{(len(uncached) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                logger.error(f"Batch embedding error: {e}")
            
            # Small delay for rate limiting
            if batch_start + batch_size < len(uncached):
                time.sleep(0.1)
        
        self._save_cache()
        return results


# ============================================================================
# LABEL GENERATOR
# ============================================================================

class LabelGenerator:
    """Generates label metadata for clusters."""
    
    def __init__(self, client: openai.OpenAI):
        self.client = client
    
    def _sanitize_id(self, text: str) -> str:
        s = text.lower().strip()
        s = re.sub(r'[^a-z0-9\s-]', '', s)
        s = re.sub(r'\s+', '-', s)
        s = re.sub(r'-+', '-', s).strip('-')
        return s[:40] if s else 'unknown'
    
    def generate_label(self, cluster_id: int, sample_claims: List[str]) -> Tuple[str, str, str, str]:
        """
        Generate label metadata from sample false claims.
        Returns: (label_id, label_name, description, severity)
        """
        claims_text = "\n".join(f"- {c}" for c in sample_claims[:5])
        
        prompt = f"""These are similar FALSE CLAIMS that spread misinformation:

{claims_text}

Create a label for this misinformation category. Respond with JSON:
{{
    "label_id": "short-id (max 30 chars, lowercase, hyphens)",
    "label_name": "Display Name (max 40 chars)",
    "description": "What this misinformation claims (max 150 chars)",
    "severity": "inform or alert"
}}

Use "alert" only for dangerous health/safety misinformation."""

        try:
            response = self.client.chat.completions.create(
                model=CONFIG.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You categorize misinformation. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = re.sub(r'^```\w*\n?', '', content)
                content = re.sub(r'\n?```$', '', content)
            
            data = json.loads(content)
            return (
                self._sanitize_id(data.get("label_id", f"misinfo-{cluster_id}")),
                data.get("label_name", f"Misinformation {cluster_id}")[:40],
                data.get("description", "Potential misinformation")[:150],
                data.get("severity", "inform") if data.get("severity") in ["inform", "alert"] else "inform"
            )
            
        except Exception as e:
            logger.error(f"Label generation failed: {e}")
            return (f"misinfo-{cluster_id}", f"Misinformation {cluster_id}", "Potential misinformation", "inform")


# ============================================================================
# MISINFORMATION ANALYZER
# ============================================================================

class MisinformationAnalyzer:
    """
    Main analyzer - matches posts against extracted false claims.
    
    Memory-bounded:
    - Clusters store only centroids (not all claims)
    - Keywords are sets with max size
    - No runtime caching of post embeddings
    """
    
    def __init__(self, embedding_service: EmbeddingService, openai_client: openai.OpenAI):
        self.embedding_service = embedding_service
        self.openai_client = openai_client
        self.claim_extractor = ClaimExtractor(openai_client)
        self.label_generator = LabelGenerator(openai_client)
        
        # Runtime data structures (bounded)
        self.clusters: List[ClaimCluster] = []
        self.label_definitions: List[LabelDefinition] = []
        self.keyword_to_clusters: Dict[str, List[int]] = {}  # Inverted index
        
        # Stats
        self.stats = {
            "posts_seen": 0,
            "keyword_matches": 0,
            "similarity_filtered": 0,
            "labeled": 0
        }
    
    def initialize(self, tsv_path: str) -> int:
        """
        Full initialization pipeline:
        1. Load Community Notes
        2. Extract false claims
        3. Embed claims
        4. Cluster similar claims
        5. Build keyword index
        
        Returns number of clusters created.
        """
        path = Path(tsv_path)
        if not path.exists():
            logger.warning(f"Notes file not found: {tsv_path}")
            return 0
        
        # Step 1: Load summaries
        logger.info("Step 1/5: Loading Community Notes...")
        summaries: List[Tuple[str, str]] = []  # (note_id, summary)
        
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if i >= CONFIG.MAX_NOTES:
                    break
                
                if row.get('classification') != 'MISINFORMED_OR_POTENTIALLY_MISLEADING':
                    continue
                
                summary = row.get('summary', '').strip()
                if len(summary) < 30:  # Skip useless short notes
                    continue
                
                # Skip notes that are just links or very short
                text_without_urls = re.sub(r'https?://\S+', '', summary).strip()
                if len(text_without_urls) < 20:
                    continue
                
                summaries.append((row.get('noteId', str(i)), summary))
        
        logger.info(f"Loaded {len(summaries)} valid Community Notes")
        
        if len(summaries) < 5:
            logger.warning("Not enough notes to process")
            return 0
        
        # Step 2: Extract false claims
        logger.info("Step 2/5: Extracting false claims...")
        extractions = self.claim_extractor.extract_claims_batch(summaries)
        
        # Build claim objects
        claims: List[ExtractedClaim] = []
        for (note_id, summary), extraction in zip(summaries, extractions):
            if extraction and extraction.get("false_claim"):
                claims.append(ExtractedClaim(
                    note_id=note_id,
                    original_summary=summary[:200],
                    false_claim=extraction["false_claim"],
                    claim_keywords=[kw.lower() for kw in extraction.get("keywords", [])]
                ))
        
        logger.info(f"Extracted {len(claims)} valid false claims")
        
        if len(claims) < 5:
            logger.warning("Not enough claims extracted")
            return 0
        
        # Step 3: Embed the FALSE CLAIMS (not the summaries!)
        logger.info("Step 3/5: Embedding false claims...")
        claim_texts = [c.false_claim for c in claims]
        embeddings = self.embedding_service.get_embeddings_batch(claim_texts)
        
        # Attach embeddings to claims
        valid_claims: List[ExtractedClaim] = []
        for claim, emb in zip(claims, embeddings):
            if emb is not None:
                claim.embedding = emb
                valid_claims.append(claim)
        
        logger.info(f"Successfully embedded {len(valid_claims)} claims")
        
        # Clear embedding cache from memory
        self.embedding_service._clear_memory()
        
        # Step 4: Cluster similar claims
        logger.info("Step 4/5: Clustering similar claims...")
        X = np.vstack([c.embedding for c in valid_claims])
        
        clustering = DBSCAN(
            eps=CONFIG.CLUSTERING_EPS,
            min_samples=CONFIG.CLUSTERING_MIN_SAMPLES,
            metric='cosine'
        ).fit(X)
        
        # Group claims by cluster
        cluster_claims: Dict[int, List[ExtractedClaim]] = defaultdict(list)
        for claim, label in zip(valid_claims, clustering.labels_):
            if label >= 0:
                cluster_claims[label].append(claim)
        
        noise_count = sum(1 for l in clustering.labels_ if l < 0)
        logger.info(f"Found {len(cluster_claims)} clusters ({noise_count} noise points)")
        
        # Step 5: Build cluster objects
        logger.info("Step 5/5: Building cluster index...")
        self.clusters = []
        
        for cluster_id in sorted(cluster_claims.keys())[:CONFIG.MAX_CLUSTERS]:
            claims_in_cluster = cluster_claims[cluster_id]
            
            # Compute centroid
            cluster_embeddings = np.vstack([c.embedding for c in claims_in_cluster])
            centroid = np.mean(cluster_embeddings, axis=0).astype(np.float32)
            
            # Collect keywords (bounded)
            all_keywords: Set[str] = set()
            for c in claims_in_cluster:
                all_keywords.update(c.claim_keywords)
                if len(all_keywords) >= CONFIG.MAX_KEYWORDS_PER_CLUSTER:
                    break
            
            # Sample claims for label generation and logging
            sample_claims = [c.false_claim for c in claims_in_cluster[:CONFIG.MAX_CLAIMS_PER_CLUSTER]]
            
            # Generate label metadata
            label_id, label_name, description, severity = \
                self.label_generator.generate_label(cluster_id, sample_claims)
            
            cluster = ClaimCluster(
                cluster_id=int(cluster_id),
                centroid=centroid,
                label_id=label_id,
                label_name=label_name,
                label_description=description,
                keywords=all_keywords,
                sample_claims=sample_claims,
                severity=severity,
                claim_count=len(claims_in_cluster)
            )
            self.clusters.append(cluster)
            
            logger.info(f"  Cluster {cluster_id}: '{label_name}' - {len(claims_in_cluster)} claims, {len(all_keywords)} keywords")
            for sc in sample_claims[:2]:
                logger.info(f"    Example: \"{sc[:80]}...\"" if len(sc) > 80 else f"    Example: \"{sc}\"")
        
        # Build inverted keyword index
        self._build_keyword_index()
        
        # Generate label definitions for Bluesky
        self._generate_label_definitions()
        
        # Memory cleanup
        del X, valid_claims, cluster_claims, embeddings, claims
        gc.collect()
        
        logger.info(f"Initialization complete: {len(self.clusters)} clusters ready")
        return len(self.clusters)
    
    def _build_keyword_index(self):
        """Build inverted index from keywords to cluster IDs."""
        self.keyword_to_clusters = defaultdict(list)
        for cluster in self.clusters:
            for keyword in cluster.keywords:
                if cluster.cluster_id not in self.keyword_to_clusters[keyword]:
                    self.keyword_to_clusters[keyword].append(cluster.cluster_id)
        
        logger.info(f"Keyword index: {len(self.keyword_to_clusters)} unique keywords")
    
    def _generate_label_definitions(self):
        """Generate Bluesky label definitions."""
        self.label_definitions = []
        for c in self.clusters:
            self.label_definitions.append(LabelDefinition(
                identifier=c.label_id,
                severity=c.severity,
                blurs="none",
                defaultSetting="warn",
                locales=[{
                    "lang": "en",
                    "name": c.label_name,
                    "description": c.label_description
                }]
            ))
        
        # Save to file for publish-labels.ts
        output = {
            "labelValues": [d.identifier for d in self.label_definitions],
            "labelValueDefinitions": [d.to_dict() for d in self.label_definitions],
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "label_id": c.label_id,
                    "label_name": c.label_name,
                    "keywords": list(c.keywords)[:10],
                    "sample_claims": c.sample_claims,
                    "claim_count": c.claim_count
                }
                for c in self.clusters
            ],
            "generated_at": datetime.now().isoformat()
        }
        
        output_path = Path(CONFIG.LABELS_OUTPUT_FILE)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved {len(self.label_definitions)} label definitions")
    
    def _find_candidate_clusters(self, text: str) -> List[ClaimCluster]:
        """Fast keyword-based pre-filter."""
        text_lower = text.lower()
        candidate_ids: Set[int] = set()
        
        for keyword, cluster_ids in self.keyword_to_clusters.items():
            # Match whole word boundaries to reduce false positives
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                candidate_ids.update(cluster_ids)
        
        return [c for c in self.clusters if c.cluster_id in candidate_ids]
    
    def analyze_post(self, text: str, author_did: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a post for misinformation.
        
        Flow:
        1. Keyword pre-filter (fast, no API call)
        2. Embed post and compare to claim centroids
        3. If similarity exceeds threshold, label it
        
        No stance detection needed - we're matching against false claims directly,
        so high similarity means the post IS making the false claim.
        """
        self.stats["posts_seen"] += 1
        
        if not text or not self.clusters:
            return None
        
        # Step 1: Keyword pre-filter
        candidates = self._find_candidate_clusters(text)
        
        if not candidates:
            return None  # No keyword match, skip silently (most posts)
        
        self.stats["keyword_matches"] += 1
        
        # Step 2: Embed the post
        post_embedding = self.embedding_service.get_embedding(text)
        if post_embedding is None:
            return None
        
        # Step 3: Find best matching cluster
        best_cluster: Optional[ClaimCluster] = None
        best_similarity: float = 0.0
        
        for cluster in candidates:
            sim = cosine_similarity(
                post_embedding.reshape(1, -1),
                cluster.centroid.reshape(1, -1)
            )[0, 0]
            
            if sim > best_similarity:
                best_similarity = sim
                best_cluster = cluster
        
        # Step 4: Check threshold
        if best_similarity < CONFIG.SIMILARITY_THRESHOLD:
            self.stats["similarity_filtered"] += 1
            logger.info(
                f"[FILTERED] similarity={best_similarity:.3f} < {CONFIG.SIMILARITY_THRESHOLD} | "
                f"cluster='{best_cluster.label_id}' | text='{text[:100]}...'"
            )
            return None
        
        # Step 5: We have a match!
        self.stats["labeled"] += 1
        is_high_confidence = best_similarity >= CONFIG.HIGH_CONFIDENCE_THRESHOLD
        confidence_marker = " ⚠️ HIGH CONFIDENCE" if is_high_confidence else ""
        
        logger.info(
            f"[MATCH] similarity={best_similarity:.3f}{confidence_marker} | "
            f"label='{best_cluster.label_id}' ({best_cluster.label_name}) | "
            f"text='{text[:100]}...'"
        )
        
        return {
            "label": best_cluster.label_id,
            "score": float(best_similarity),
            "cluster_id": best_cluster.cluster_id,
            "theme": best_cluster.label_name,
            "severity": best_cluster.severity,
            "high_confidence": is_high_confidence,
            "text": text[:200]
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            **self.stats,
            "clusters": len(self.clusters),
            "keywords": len(self.keyword_to_clusters),
            "label_rate": f"{(self.stats['labeled'] / max(1, self.stats['keyword_matches']) * 100):.2f}%" 
                if self.stats['keyword_matches'] > 0 else "N/A"
        }


# ============================================================================
# GLOBAL STATE
# ============================================================================

class AnalyzerState:
    analyzer: Optional[MisinformationAnalyzer] = None
    initialized: bool = False
    init_error: Optional[str] = None

state = AnalyzerState()


def initialize():
    """Initialize the analyzer."""
    logger.info("=" * 60)
    logger.info("MISINFORMATION ANALYZER v3 - CLAIM-BASED MATCHING")
    logger.info("=" * 60)
    
    errors = CONFIG.validate()
    if errors:
        state.init_error = "; ".join(errors)
        logger.error(f"Config errors: {state.init_error}")
        return False
    
    try:
        client = openai.OpenAI(api_key=CONFIG.OPENAI_API_KEY)
        embedding_service = EmbeddingService(CONFIG.OPENAI_API_KEY)
        
        analyzer = MisinformationAnalyzer(embedding_service, client)
        num_clusters = analyzer.initialize(CONFIG.NOTES_FILE)
        
        if num_clusters > 0:
            state.analyzer = analyzer
            state.initialized = True
            
            logger.info("-" * 40)
            logger.info("ACTIVE LABELS:")
            for c in analyzer.clusters:
                logger.info(f"  • {c.label_id}: {c.label_name}")
                logger.info(f"    Keywords: {', '.join(list(c.keywords)[:5])}...")
            logger.info("-" * 40)
        else:
            state.init_error = "No clusters created"
            logger.warning("Analyzer initialized but no clusters created")
        
        gc.collect()
        logger.info("=" * 60)
        logger.info("INITIALIZATION COMPLETE")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        state.init_error = str(e)
        logger.exception(f"Initialization failed: {e}")
        return False


# ============================================================================
# FASTAPI APP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize()
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="Misinformation Analyzer v3",
    version="3.0.0",
    lifespan=lifespan
)


class PostPayload(BaseModel):
    text: str
    did: str


@app.post("/analyze")
async def analyze(payload: PostPayload):
    if not state.initialized or not state.analyzer:
        return {"label": None, "error": state.init_error or "Not initialized"}
    
    if not payload.text:
        return {"label": None}
    
    result = state.analyzer.analyze_post(payload.text, payload.did)
    return result if result else {"label": None}


@app.get("/health")
async def health():
    return {
        "status": "healthy" if state.initialized else "initializing",
        "initialized": state.initialized,
        "error": state.init_error,
        "clusters": len(state.analyzer.clusters) if state.analyzer else 0
    }


@app.get("/stats")
async def stats():
    if not state.analyzer:
        return {"error": "Not initialized"}
    return state.analyzer.get_stats()


@app.get("/labels")
async def get_labels():
    if not state.analyzer:
        return {"error": "Not initialized"}
    return {
        "labelValues": [d.identifier for d in state.analyzer.label_definitions],
        "labelValueDefinitions": [d.to_dict() for d in state.analyzer.label_definitions]
    }


@app.get("/clusters")
async def get_clusters():
    """Debug endpoint to see what claims we're matching against."""
    if not state.analyzer:
        return {"error": "Not initialized"}
    return {
        "clusters": [
            {
                "id": c.cluster_id,
                "label": c.label_id,
                "name": c.label_name,
                "description": c.label_description,
                "keywords": list(c.keywords),
                "sample_claims": c.sample_claims,
                "claim_count": c.claim_count
            }
            for c in state.analyzer.clusters
        ]
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="info")