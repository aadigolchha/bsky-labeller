import os
import re
import json
import hashlib
import logging
import pickle
import csv
import gc
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from contextlib import asynccontextmanager

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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration with sensible defaults, overridable via environment variables."""
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
    LLM_MODEL: str = os.getenv('LLM_MODEL', 'gpt-4o-mini')
    EMBEDDING_BATCH_SIZE: int = int(os.getenv('EMBEDDING_BATCH_SIZE', '100'))
    
    # Clustering
    CLUSTERING_EPS: float = float(os.getenv('CLUSTERING_EPS', '0.3'))
    CLUSTERING_MIN_SAMPLES: int = int(os.getenv('CLUSTERING_MIN_SAMPLES', '2'))
    
    # Detection thresholds
    SIMILARITY_THRESHOLD: float = float(os.getenv('SIMILARITY_THRESHOLD', '0.45'))
    HIGH_CONFIDENCE_THRESHOLD: float = float(os.getenv('HIGH_CONFIDENCE_THRESHOLD', '0.70'))
    
    # Data paths (production uses /data volume)
    IS_PRODUCTION: bool = os.getenv('NODE_ENV') == 'production'
    DATA_DIR: str = '/data' if IS_PRODUCTION else './data'
    CACHE_DIR: str = f"{DATA_DIR}/cache"
    NOTES_FILE: str = os.getenv('NOTES_FILE', f"{DATA_DIR}/notes-00000.tsv")
    LABELS_OUTPUT_FILE: str = f"{DATA_DIR}/generated-labels.json"
    
    # Limits - REDUCED for memory efficiency
    MAX_NOTES: int = int(os.getenv('MAX_NOTES', '500'))  # Reduced from 1000
    MAX_CLUSTERS: int = int(os.getenv('MAX_CLUSTERS', '30'))  # Reduced from 50
    
    @classmethod
    def validate(cls) -> List[str]:
        """Validate configuration, return list of errors."""
        errors = []
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
        return errors

CONFIG = Config()

# ============================================================================
# DATA CLASSES (Memory-Optimized)
# ============================================================================

@dataclass
class LightweightCluster:
    """
    Memory-efficient cluster representation.
    Only stores what's needed for runtime matching - no raw notes.
    """
    cluster_id: int
    centroid: np.ndarray  # The only large object we keep
    label_id: str
    label_name: str
    label_description: str
    keywords: List[str]
    severity: str = "inform"
    blurs: str = "none"
    note_count: int = 0  # Just the count, not the notes themselves
    sample_summaries: List[str] = field(default_factory=list)  # 2-3 examples for stance detection


@dataclass 
class LabelDefinition:
    """A label definition ready for Bluesky registration."""
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
# EMBEDDING SERVICE (Disk-Only Cache)
# ============================================================================

class EmbeddingService:
    """
    Handles embedding generation with DISK-ONLY caching.
    No in-memory cache to save RAM.
    """
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = CONFIG.EMBEDDING_MODEL
        self.cache_dir = Path(CONFIG.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._disk_cache: Dict[str, np.ndarray] = {}
        self._load_disk_cache()
    
    def _cache_path(self) -> Path:
        return self.cache_dir / 'embeddings.pkl'
    
    def _load_disk_cache(self):
        """Load cached embeddings from disk into temporary dict for batch processing."""
        if self._cache_path().exists():
            try:
                with open(self._cache_path(), 'rb') as f:
                    self._disk_cache = pickle.load(f)
                logger.info(f"Loaded {len(self._disk_cache)} cached embeddings from disk")
            except Exception as e:
                logger.warning(f"Could not load embedding cache: {e}")
                self._disk_cache = {}
    
    def _save_disk_cache(self):
        """Persist embedding cache to disk."""
        try:
            with open(self._cache_path(), 'wb') as f:
                pickle.dump(self._disk_cache, f)
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")
    
    def _clear_memory_cache(self):
        """Clear the in-memory cache after initialization to free RAM."""
        self._disk_cache = {}
        gc.collect()
    
    def _text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:32]
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a single text - always calls API (no memory cache)."""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Embedding API error: {e}")
            return None
    
    def get_embeddings_batch_for_init(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        Get embeddings for initialization. Uses disk cache, then clears memory.
        Only used during startup.
        """
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []
        
        # Check disk cache
        for i, text in enumerate(texts):
            text_hash = self._text_hash(text)
            if text_hash in self._disk_cache:
                results[i] = self._disk_cache[text_hash]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        # Batch API calls for uncached texts
        if uncached_texts:
            batch_size = CONFIG.EMBEDDING_BATCH_SIZE
            for batch_start in range(0, len(uncached_texts), batch_size):
                batch_end = min(batch_start + batch_size, len(uncached_texts))
                batch = uncached_texts[batch_start:batch_end]
                batch_indices = uncached_indices[batch_start:batch_end]
                
                try:
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.model
                    )
                    
                    for j, emb_data in enumerate(response.data):
                        embedding = np.array(emb_data.embedding, dtype=np.float32)
                        original_idx = batch_indices[j]
                        results[original_idx] = embedding
                        
                        # Save to disk cache
                        text_hash = self._text_hash(texts[original_idx])
                        self._disk_cache[text_hash] = embedding
                    
                    logger.info(f"Embedded batch {batch_start//batch_size + 1}: {len(batch)} texts")
                    
                except Exception as e:
                    logger.error(f"Batch embedding error: {e}")
            
            self._save_disk_cache()
        
        return results


# ============================================================================
# LABEL GENERATOR
# ============================================================================

class LabelGenerator:
    """Generates Bluesky label definitions from misinformation clusters."""
    
    def __init__(self, openai_client: openai.OpenAI):
        self.client = openai_client
    
    def sanitize_label_id(self, text: str) -> str:
        """Convert text to valid Bluesky label identifier."""
        sanitized = text.lower().strip()
        sanitized = re.sub(r'[^a-z0-9\s-]', '', sanitized)
        sanitized = re.sub(r'\s+', '-', sanitized)
        sanitized = re.sub(r'-+', '-', sanitized)
        sanitized = sanitized.strip('-')
        return sanitized[:50] if sanitized else 'unknown'
    
    def generate_label_for_cluster(
        self, 
        cluster_id: int, 
        sample_summaries: List[str]
    ) -> Tuple[str, str, str, str, List[str]]:
        """
        Use LLM to generate a meaningful label for a cluster of misinformation.
        
        Returns: (label_id, label_name, description, severity, keywords)
        """
        summaries_text = "\n".join(f"- {s}" for s in sample_summaries[:5])
        
        prompt = f"""Analyze these Community Notes that fact-check similar misinformation:

{summaries_text}

Based on these notes, create a label for this type of misinformation. Respond with JSON:
{{
    "label_id": "short-hyphenated-id (max 30 chars, lowercase, e.g., 'covid-vaccine-myths')",
    "label_name": "Human-readable name (max 50 chars, e.g., 'COVID Vaccine Misinformation')",
    "description": "Clear description of this misinformation type (max 200 chars)",
    "severity": "inform or alert (use 'alert' only for dangerous health/safety misinformation)",
    "keywords": ["list", "of", "5-10", "relevant", "keywords"]
}}

Important:
- label_id must be lowercase, alphanumeric with hyphens only
- Be specific about what type of misinformation this is
- Keywords should help identify similar posts"""

        try:
            response = self.client.chat.completions.create(
                model=CONFIG.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing misinformation patterns. Generate clear, accurate labels for fact-checking purposes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            label_id = self.sanitize_label_id(result.get('label_id', f'misinfo-{cluster_id}'))
            label_name = result.get('label_name', f'Misinformation Type {cluster_id}')[:50]
            description = result.get('description', 'Potential misinformation identified by Community Notes.')[:200]
            severity = result.get('severity', 'inform')
            if severity not in ['inform', 'alert', 'none']:
                severity = 'inform'
            keywords = result.get('keywords', [])[:10]
            
            return label_id, label_name, description, severity, keywords
            
        except Exception as e:
            logger.error(f"Label generation failed for cluster {cluster_id}: {e}")
            return (
                f"misinfo-{cluster_id}",
                f"Misinformation Pattern {cluster_id}",
                "Content flagged by Community Notes analysis.",
                "inform",
                []
            )


# ============================================================================
# MISINFORMATION ANALYZER (Memory-Optimized)
# ============================================================================

class MisinformationAnalyzer:
    """
    Main analyzer - optimized for low memory usage.
    Only keeps centroids and keywords in memory after initialization.
    """
    
    def __init__(self, embedding_service: EmbeddingService, openai_client: openai.OpenAI):
        self.embedding_service = embedding_service
        self.openai_client = openai_client
        self.label_generator = LabelGenerator(openai_client)
        self.clusters: List[LightweightCluster] = []
        self.label_definitions: List[LabelDefinition] = []
        self.keyword_index: Dict[str, List[int]] = {}
    
    def load_and_process_notes(self, tsv_path: str) -> int:
        """
        Load Community Notes, cluster them, then DISCARD the raw data.
        Returns number of clusters created.
        """
        path = Path(tsv_path)
        
        if not path.exists():
            logger.warning(f"Community Notes file not found: {tsv_path}")
            return 0
        
        logger.info(f"Loading Community Notes from {tsv_path}")
        
        # Temporary storage - will be cleared after clustering
        summaries: List[str] = []
        
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            for i, row in enumerate(reader):
                if CONFIG.MAX_NOTES and i >= CONFIG.MAX_NOTES:
                    break
                
                classification = row.get('classification', '')
                summary = row.get('summary', '')
                
                if classification != 'MISINFORMED_OR_POTENTIALLY_MISLEADING':
                    continue
                if not summary or len(summary) < 20:
                    continue
                
                summaries.append(summary)
        
        logger.info(f"Loaded {len(summaries)} valid Community Notes summaries")
        
        if len(summaries) < CONFIG.CLUSTERING_MIN_SAMPLES:
            logger.warning(f"Not enough notes for clustering: {len(summaries)}")
            return 0
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(summaries)} notes...")
        embeddings = self.embedding_service.get_embeddings_batch_for_init(summaries)
        
        # Filter out failed embeddings
        valid_pairs = [(s, e) for s, e in zip(summaries, embeddings) if e is not None]
        if not valid_pairs:
            logger.error("No valid embeddings generated")
            return 0
        
        valid_summaries, valid_embeddings = zip(*valid_pairs)
        valid_summaries = list(valid_summaries)
        
        logger.info(f"Successfully embedded {len(valid_embeddings)} notes")
        
        # Build embedding matrix
        X = np.vstack(valid_embeddings)
        
        # Clear the disk cache from memory now that we have embeddings
        self.embedding_service._clear_memory_cache()
        gc.collect()
        
        # Cluster
        logger.info("Clustering notes...")
        clustering = DBSCAN(
            eps=CONFIG.CLUSTERING_EPS,
            min_samples=CONFIG.CLUSTERING_MIN_SAMPLES,
            metric='cosine'
        ).fit(X)
        
        # Group by cluster
        cluster_data: Dict[int, Dict] = defaultdict(lambda: {
            'indices': [],
            'embeddings': [],
            'summaries': []
        })
        
        for idx, label in enumerate(clustering.labels_):
            if label >= 0:  # -1 is noise
                cluster_data[label]['indices'].append(idx)
                cluster_data[label]['embeddings'].append(X[idx])
                cluster_data[label]['summaries'].append(valid_summaries[idx])
        
        logger.info(f"Found {len(cluster_data)} clusters")
        
        # Build lightweight clusters
        self.clusters = []
        for cluster_id in sorted(cluster_data.keys())[:CONFIG.MAX_CLUSTERS]:
            cluster_id = int(cluster_id)
            data = cluster_data[cluster_id]
            
            # Compute centroid
            centroid = np.mean(np.vstack(data['embeddings']), axis=0).astype(np.float32)
            
            # Keep only 3 sample summaries for stance detection context
            sample_summaries = data['summaries'][:3]
            
            # Generate label
            label_id, label_name, description, severity, keywords = \
                self.label_generator.generate_label_for_cluster(cluster_id, data['summaries'])
            
            cluster = LightweightCluster(
                cluster_id=cluster_id,
                centroid=centroid,
                label_id=label_id,
                label_name=label_name,
                label_description=description,
                keywords=keywords,
                severity=severity,
                note_count=len(data['indices']),
                sample_summaries=sample_summaries
            )
            self.clusters.append(cluster)
            
            logger.info(f"Cluster {cluster_id}: '{label_name}' ({len(data['indices'])} notes)")
        
        # Build keyword index
        self._build_keyword_index()
        
        # Generate label definitions
        self._generate_label_definitions()
        
        # CRITICAL: Clear all temporary data
        del X, valid_embeddings, valid_summaries, summaries, cluster_data, clustering
        gc.collect()
        
        logger.info(f"Memory cleanup complete. Keeping {len(self.clusters)} cluster centroids.")
        
        return len(self.clusters)
    
    def _build_keyword_index(self):
        """Build an inverted index from keywords to cluster IDs."""
        self.keyword_index = defaultdict(list)
        for cluster in self.clusters:
            for keyword in cluster.keywords:
                kw_lower = keyword.lower()
                if cluster.cluster_id not in self.keyword_index[kw_lower]:
                    self.keyword_index[kw_lower].append(cluster.cluster_id)
        
        logger.info(f"Built keyword index with {len(self.keyword_index)} unique keywords")
    
    def _generate_label_definitions(self):
        """Generate Bluesky label definitions and save to file."""
        self.label_definitions = []
        for c in self.clusters:
            self.label_definitions.append(LabelDefinition(
                identifier=c.label_id,
                severity=c.severity,
                blurs=c.blurs,
                defaultSetting="warn",
                locales=[{
                    "lang": "en",
                    "name": c.label_name,
                    "description": c.label_description
                }]
            ))
        
        # Save to JSON
        output = {
            "labelValues": [d.identifier for d in self.label_definitions],
            "labelValueDefinitions": [d.to_dict() for d in self.label_definitions],
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "label_id": c.label_id,
                    "theme": c.label_name,
                    "keywords": c.keywords,
                    "note_count": c.note_count
                }
                for c in self.clusters
            ],
            "generated_at": datetime.now().isoformat()
        }
        
        output_path = Path(CONFIG.LABELS_OUTPUT_FILE)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved {len(self.label_definitions)} label definitions to {output_path}")
    
    def get_candidate_clusters(self, text: str) -> List[LightweightCluster]:
        """Use keyword index for fast pre-filtering."""
        if not self.keyword_index:
            return self.clusters
        
        text_lower = text.lower()
        candidate_ids = set()
        
        for keyword, cluster_ids in self.keyword_index.items():
            if keyword in text_lower:
                candidate_ids.update(cluster_ids)
        
        if not candidate_ids:
            return []
        
        return [c for c in self.clusters if c.cluster_id in candidate_ids]
    
    def detect_stance(self, text: str, cluster: LightweightCluster) -> str:
        """Determine if the text supports, refutes, or neutrally discusses the misinformation."""
        prompt = f"""Analyze the stance of this social media post towards a known misinformation theme.

Misinformation theme: {cluster.label_name}
Description: {cluster.label_description}

Post text:
"{text}"

Does this post:
1. SUPPORT the misinformation (spreading it, agreeing with it)
2. REFUTE the misinformation (fact-checking, correcting, debunking)
3. NEUTRAL (just discussing the topic without taking a clear stance)

Respond with exactly one word: supports, refutes, or neutral"""

        try:
            response = self.openai_client.chat.completions.create(
                model=CONFIG.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert at stance detection for misinformation analysis. Be conservative - only return 'supports' if the post is clearly spreading misinformation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            stance = response.choices[0].message.content.strip().lower()
            if stance in ['supports', 'refutes', 'neutral']:
                return stance
            return 'neutral'
            
        except Exception as e:
            logger.error(f"Stance detection failed: {e}")
            return 'neutral'
    
    def analyze_post(self, text: str, author_did: str) -> Optional[Dict[str, Any]]:
        """Analyze a post for misinformation."""
        if not text or not self.clusters:
            return None
        
        # Fast keyword pre-filter
        candidates = self.get_candidate_clusters(text)
        if not candidates:
            return None
        
        # Generate embedding for the post
        post_embedding = self.embedding_service.get_embedding(text)
        if post_embedding is None:
            return None
        
        # Find best matching cluster
        best_cluster = None
        best_similarity = 0.0
        
        for cluster in candidates:
            similarity = cosine_similarity(
                post_embedding.reshape(1, -1),
                cluster.centroid.reshape(1, -1)
            )[0, 0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster
        
        # Check threshold
        if best_similarity < CONFIG.SIMILARITY_THRESHOLD:
            return None
        
        # Stance detection
        stance = self.detect_stance(text, best_cluster)
        
        if stance != 'supports':
            logger.info(f"[STANCE] Matched '{best_cluster.label_id}' ({best_similarity:.2f}) but stance='{stance}'")
            return None
        
        is_high_confidence = best_similarity >= CONFIG.HIGH_CONFIDENCE_THRESHOLD
        
        return {
            "label": best_cluster.label_id,
            "score": float(best_similarity),
            "cluster_id": best_cluster.cluster_id,
            "theme": best_cluster.label_name,
            "severity": best_cluster.severity,
            "high_confidence": is_high_confidence,
            "text": text[:200]
        }


# ============================================================================
# GLOBAL STATE
# ============================================================================

class AnalyzerState:
    """Global application state."""
    analyzer: Optional[MisinformationAnalyzer] = None
    initialized: bool = False
    init_error: Optional[str] = None
    label_count: int = 0

state = AnalyzerState()


def initialize():
    """Initialize the analyzer with Community Notes data."""
    logger.info("=" * 60)
    logger.info("INITIALIZING MISINFORMATION ANALYZER (Memory-Optimized)")
    logger.info("=" * 60)
    
    errors = CONFIG.validate()
    if errors:
        state.init_error = "; ".join(errors)
        logger.error(f"Configuration errors: {state.init_error}")
        return False
    
    try:
        openai_client = openai.OpenAI(api_key=CONFIG.OPENAI_API_KEY)
        embedding_service = EmbeddingService(CONFIG.OPENAI_API_KEY)
        analyzer = MisinformationAnalyzer(embedding_service, openai_client)
        
        num_clusters = analyzer.load_and_process_notes(CONFIG.NOTES_FILE)
        
        if num_clusters > 0:
            state.label_count = num_clusters
            logger.info(f"Generated {num_clusters} dynamic labels")
            
            logger.info("-" * 40)
            logger.info("REGISTERED LABELS:")
            for c in analyzer.clusters[:10]:
                logger.info(f"  - {c.label_id}: {c.label_name}")
            if len(analyzer.clusters) > 10:
                logger.info(f"  ... and {len(analyzer.clusters) - 10} more")
            logger.info("-" * 40)
        else:
            logger.warning("No clusters created from Community Notes")
        
        state.analyzer = analyzer
        state.initialized = True
        
        # Force garbage collection
        gc.collect()
        
        logger.info("=" * 60)
        logger.info("ANALYZER INITIALIZATION COMPLETE")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        state.init_error = str(e)
        logger.exception(f"Initialization failed: {e}")
        return False


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    initialize()
    yield
    logger.info("Shutting down analyzer...")

app = FastAPI(
    title="Bluesky Misinformation Analyzer",
    description="Analyzes posts for misinformation using Community Notes data",
    version="2.0.0",
    lifespan=lifespan
)


class PostPayload(BaseModel):
    """Request payload for post analysis."""
    text: str
    did: str


@app.post("/analyze")
async def analyze(payload: PostPayload):
    """Analyze a post for misinformation."""
    if not state.initialized or state.analyzer is None:
        return {"label": None, "error": state.init_error or "Not initialized"}
    
    if not payload.text:
        return {"label": None}
    
    result = state.analyzer.analyze_post(payload.text, payload.did)
    
    if result:
        return result
    
    return {"label": None}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if state.initialized else "initializing",
        "initialized": state.initialized,
        "error": state.init_error,
        "clusters": len(state.analyzer.clusters) if state.analyzer else 0,
        "labels": state.label_count
    }


@app.get("/stats")
async def stats():
    """Get analyzer statistics."""
    if not state.analyzer:
        return {"error": "Not initialized"}
    
    return {
        "clusters": len(state.analyzer.clusters),
        "keywords": len(state.analyzer.keyword_index),
        "labels": [
            {
                "id": c.label_id,
                "name": c.label_name,
                "description": c.label_description,
                "severity": c.severity,
                "keywords": c.keywords[:5],
                "note_count": c.note_count
            }
            for c in state.analyzer.clusters
        ]
    }


@app.get("/labels")
async def get_labels():
    """Get the generated label definitions."""
    if not state.analyzer:
        return {"error": "Not initialized"}
    
    return {
        "labelValues": [d.identifier for d in state.analyzer.label_definitions],
        "labelValueDefinitions": [d.to_dict() for d in state.analyzer.label_definitions]
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5001,
        log_level="info"
    )