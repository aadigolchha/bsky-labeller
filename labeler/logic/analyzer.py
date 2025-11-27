"""
Bluesky Labeler - Misinformation & AI-Generated Content Detector
=================================================================

Uses Community Notes data to identify and label content on Bluesky.
Returns one of two labels: "misinformation" or "ai-generated", or None.

This analyzer:
1. Loads Community Notes TSV data
2. Generates embeddings and clusters similar misinformation themes
3. Analyzes incoming posts for similarity to known misinformation patterns
4. Uses LLM-based stance detection to avoid labeling counter-speech
5. Returns "ai-generated" if the matched cluster relates to AI/manipulated media
6. Returns "misinformation" for other types of misinformation

Integration:
- Runs as a FastAPI server on Unix Domain Socket
- Called by Node.js jetstream.ts for each incoming post
- Returns {label: string | null, score?: float, ...} format
"""

import os
import re
import json
import hashlib
import logging
import pickle
import csv
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
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

    # Limits
    MAX_NOTES: int = int(os.getenv('MAX_NOTES', '1000'))
    MAX_CLUSTERS: int = int(os.getenv('MAX_CLUSTERS', '50'))

    @classmethod
    def validate(cls) -> List[str]:
        """Validate configuration, return list of errors."""
        errors = []
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
        return errors

CONFIG = Config()

# ============================================================================
# FIXED LABELS - Only these two labels are used
# ============================================================================

LABEL_MISINFORMATION = "misinformation"
LABEL_AI_GENERATED = "ai-generated"

# Keywords that indicate AI-generated/manipulated content
AI_KEYWORDS = [
    'ai-generated', 'ai generated', 'artificial intelligence',
    'deepfake', 'deep fake', 'manipulated media', 'manipulated image',
    'manipulated video', 'fake image', 'fake video', 'synthetic',
    'generated image', 'generated video', 'photoshop', 'edited',
    'not real', 'fabricated', 'digitally altered', 'digitally created',
    'computer generated', 'cgi', 'fake photo', 'altered photo',
    'doctored', 'midjourney', 'dall-e', 'stable diffusion'
]

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CommunityNote:
    """A Community Note from X/Twitter that identifies misinformation."""
    note_id: str
    tweet_id: str
    classification: str
    summary: str
    misleading_type: str
    created_at: datetime
    embedding: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None
    is_ai_related: bool = False  # Flag for AI/manipulated media content


@dataclass
class MisinfoCluster:
    """A cluster of semantically similar misinformation."""
    cluster_id: int
    notes: List[CommunityNote]
    centroid: np.ndarray
    theme: str
    keywords: List[str]
    is_ai_related: bool = False  # True if cluster is about AI-generated/manipulated content


# ============================================================================
# EMBEDDING SERVICE
# ============================================================================

class EmbeddingService:
    """Handles embedding generation with caching."""

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = CONFIG.EMBEDDING_MODEL
        self.cache_dir = Path(CONFIG.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, np.ndarray] = {}
        self._load_cache()

    def _cache_path(self) -> Path:
        return self.cache_dir / 'embeddings.pkl'

    def _load_cache(self):
        """Load cached embeddings from disk."""
        if self._cache_path().exists():
            try:
                with open(self._cache_path(), 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded {len(self.cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Could not load embedding cache: {e}")
                self.cache = {}

    def _save_cache(self):
        """Persist embedding cache to disk."""
        try:
            with open(self._cache_path(), 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")

    def _text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:32]

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a single text, using cache if available."""
        text_hash = self._text_hash(text)

        if text_hash in self.cache:
            return self.cache[text_hash]

        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            embedding = np.array(response.data[0].embedding)
            self.cache[text_hash] = embedding
            # Lazy cache save - every 100 new embeddings
            if len(self.cache) % 100 == 0:
                self._save_cache()
            return embedding
        except Exception as e:
            logger.error(f"Embedding API error: {e}")
            return None

    def get_embeddings_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Get embeddings for multiple texts efficiently."""
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # Check cache first
        for i, text in enumerate(texts):
            text_hash = self._text_hash(text)
            if text_hash in self.cache:
                results[i] = self.cache[text_hash]
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
                        embedding = np.array(emb_data.embedding)
                        original_idx = batch_indices[j]
                        results[original_idx] = embedding

                        # Cache it
                        text_hash = self._text_hash(texts[original_idx])
                        self.cache[text_hash] = embedding

                    logger.info(f"Embedded batch {batch_start//batch_size + 1}: {len(batch)} texts")

                except Exception as e:
                    logger.error(f"Batch embedding error: {e}")

            self._save_cache()

        return results


# ============================================================================
# MISINFORMATION ANALYZER
# ============================================================================

class MisinformationAnalyzer:
    """Main analyzer that processes Community Notes and analyzes posts."""

    def __init__(self, embedding_service: EmbeddingService, openai_client: openai.OpenAI):
        self.embedding_service = embedding_service
        self.openai_client = openai_client
        self.clusters: List[MisinfoCluster] = []
        self.keyword_index: Dict[str, List[int]] = {}  # keyword -> cluster_ids

    def _is_ai_related(self, text: str) -> bool:
        """Check if text relates to AI-generated or manipulated media content."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in AI_KEYWORDS)

    def load_community_notes(self, tsv_path: str) -> List[CommunityNote]:
        """Load and filter Community Notes from TSV file."""
        notes = []
        path = Path(tsv_path)

        if not path.exists():
            logger.warning(f"Community Notes file not found: {tsv_path}")
            return notes

        logger.info(f"Loading Community Notes from {tsv_path}")

        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')

            for i, row in enumerate(reader):
                if CONFIG.MAX_NOTES and i >= CONFIG.MAX_NOTES:
                    break

                # Only include notes that identify misinformation
                classification = row.get('classification', '')
                summary = row.get('summary', '')

                if classification != 'MISINFORMED_OR_POTENTIALLY_MISLEADING':
                    continue
                if not summary or len(summary) < 20:
                    continue

                # Categorize the type of misinformation
                misleading_types = []
                is_manipulated_media = row.get('misleadingManipulatedMedia') == '1'

                for field, label in [
                    ('misleadingFactualError', 'factual_error'),
                    ('misleadingManipulatedMedia', 'manipulated_media'),
                    ('misleadingMissingImportantContext', 'missing_context'),
                    ('misleadingUnverifiedClaimAsFact', 'unverified_claim'),
                    ('misleadingSatire', 'satire'),
                    ('misleadingOutdatedInformation', 'outdated'),
                ]:
                    if row.get(field) == '1':
                        misleading_types.append(label)

                try:
                    created_at = datetime.fromtimestamp(int(row.get('createdAtMillis', 0)) / 1000)
                except (ValueError, TypeError):
                    created_at = datetime.now()

                # Check if this note is about AI-generated content
                is_ai = is_manipulated_media or self._is_ai_related(summary)

                note = CommunityNote(
                    note_id=row.get('noteId', str(i)),
                    tweet_id=row.get('tweetId', ''),
                    classification=classification,
                    summary=summary,
                    misleading_type=','.join(misleading_types) if misleading_types else 'general',
                    created_at=created_at,
                    is_ai_related=is_ai
                )
                notes.append(note)

        ai_count = sum(1 for n in notes if n.is_ai_related)
        logger.info(f"Loaded {len(notes)} valid Community Notes ({ai_count} AI-related)")
        return notes

    def generate_embeddings(self, notes: List[CommunityNote]):
        """Generate embeddings for all notes."""
        logger.info(f"Generating embeddings for {len(notes)} notes...")

        texts = [note.summary for note in notes]
        embeddings = self.embedding_service.get_embeddings_batch(texts)

        for note, embedding in zip(notes, embeddings):
            note.embedding = embedding

        valid = sum(1 for n in notes if n.embedding is not None)
        logger.info(f"Successfully embedded {valid}/{len(notes)} notes")

    def cluster_notes(self, notes: List[CommunityNote]) -> List[MisinfoCluster]:
        """Cluster notes using DBSCAN."""
        notes_with_embeddings = [n for n in notes if n.embedding is not None]

        if len(notes_with_embeddings) < CONFIG.CLUSTERING_MIN_SAMPLES:
            logger.warning(f"Not enough notes for clustering: {len(notes_with_embeddings)}")
            return []

        logger.info(f"Clustering {len(notes_with_embeddings)} notes...")

        # Build embedding matrix
        X = np.vstack([n.embedding for n in notes_with_embeddings])

        # DBSCAN clustering with cosine distance
        clustering = DBSCAN(
            eps=CONFIG.CLUSTERING_EPS,
            min_samples=CONFIG.CLUSTERING_MIN_SAMPLES,
            metric='cosine'
        ).fit(X)

        # Group notes by cluster
        cluster_notes_map: Dict[int, List[CommunityNote]] = defaultdict(list)
        for note, label in zip(notes_with_embeddings, clustering.labels_):
            if label >= 0:  # -1 is noise
                note.cluster_id = label
                cluster_notes_map[label].append(note)

        logger.info(f"Found {len(cluster_notes_map)} clusters (+ {sum(1 for l in clustering.labels_ if l == -1)} noise points)")

        # Build MisinfoCluster objects
        clusters = []
        for cluster_id, notes_in_cluster in sorted(cluster_notes_map.items())[:CONFIG.MAX_CLUSTERS]:
            cluster_id = int(cluster_id)

            # Compute centroid
            embeddings = np.vstack([n.embedding for n in notes_in_cluster])
            centroid = np.mean(embeddings, axis=0)

            # Determine if cluster is AI-related (majority vote)
            ai_count = sum(1 for n in notes_in_cluster if n.is_ai_related)
            is_ai_cluster = ai_count > len(notes_in_cluster) / 2

            # Extract keywords from summaries
            keywords = self._extract_keywords(notes_in_cluster)

            # Generate theme description
            theme = self._generate_theme(notes_in_cluster)

            cluster = MisinfoCluster(
                cluster_id=cluster_id,
                notes=notes_in_cluster,
                centroid=centroid,
                theme=theme,
                keywords=keywords,
                is_ai_related=is_ai_cluster
            )
            clusters.append(cluster)

            label_type = LABEL_AI_GENERATED if is_ai_cluster else LABEL_MISINFORMATION
            logger.info(f"Cluster {cluster_id}: '{theme[:50]}' ({len(notes_in_cluster)} notes) -> {label_type}")

        self.clusters = clusters
        self._build_keyword_index()

        return clusters

    def _extract_keywords(self, notes: List[CommunityNote]) -> List[str]:
        """Extract common keywords from cluster notes."""
        # Simple keyword extraction from summaries
        word_counts: Dict[str, int] = defaultdict(int)
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'this', 'that', 'it', 'to', 'of', 'and', 'in', 'for', 'on', 'with', 'as', 'by', 'at', 'from', 'or', 'be', 'not', 'has', 'have', 'had', 'but', 'they', 'their', 'there', 'these', 'those', 'which', 'who', 'what', 'when', 'where', 'why', 'how'}

        for note in notes:
            words = re.findall(r'\b[a-zA-Z]{4,}\b', note.summary.lower())
            for word in words:
                if word not in stop_words:
                    word_counts[word] += 1

        # Return top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:10] if count >= 2]

    def _generate_theme(self, notes: List[CommunityNote]) -> str:
        """Generate a theme description for the cluster."""
        # Use first note's summary as base theme (simplified)
        if notes:
            summary = notes[0].summary
            # Truncate to first sentence or 100 chars
            if '. ' in summary:
                return summary.split('. ')[0]
            return summary[:100]
        return "Unknown theme"

    def _build_keyword_index(self):
        """Build an inverted index from keywords to cluster IDs for fast filtering."""
        self.keyword_index = defaultdict(list)
        for cluster in self.clusters:
            for keyword in cluster.keywords:
                kw_lower = keyword.lower()
                if cluster.cluster_id not in self.keyword_index[kw_lower]:
                    self.keyword_index[kw_lower].append(cluster.cluster_id)

        logger.info(f"Built keyword index with {len(self.keyword_index)} unique keywords")

    def get_candidate_clusters(self, text: str) -> List[MisinfoCluster]:
        """Use keyword index for fast pre-filtering of potentially relevant clusters."""
        if not self.keyword_index:
            return self.clusters  # No index, check all

        text_lower = text.lower()
        candidate_ids = set()

        for keyword, cluster_ids in self.keyword_index.items():
            if keyword in text_lower:
                candidate_ids.update(cluster_ids)

        if not candidate_ids:
            return []  # No keyword matches

        return [c for c in self.clusters if c.cluster_id in candidate_ids]

    def detect_stance(self, text: str, theme: str) -> str:
        """
        Determine if the text supports, refutes, or neutrally discusses the misinformation.

        This is critical to avoid labeling counter-speech and fact-checks.
        """
        prompt = f"""Analyze the stance of this social media post towards a known misinformation theme.

Misinformation theme: {theme}

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
            return 'neutral'  # Safe default

    def analyze_post(self, text: str, author_did: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a post for misinformation or AI-generated content.

        Returns:
            - {"label": "misinformation", ...} for general misinformation
            - {"label": "ai-generated", ...} for AI/manipulated media content
            - None if no issues detected
        """
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

        # Check if similarity exceeds threshold
        if best_similarity < CONFIG.SIMILARITY_THRESHOLD:
            logger.info(f"[FILTERED] Low similarity ({best_similarity:.2f}): '{text[:80]}...'")
            return None

        # Stance detection - only label if post supports the misinformation
        stance = self.detect_stance(text, best_cluster.theme)

        if stance != 'supports':
            logger.info(f"[STANCE] Matched but stance='{stance}', not labeling: '{text[:80]}...'")
            return None

        # Determine which label to use based on cluster type
        label = LABEL_AI_GENERATED if best_cluster.is_ai_related else LABEL_MISINFORMATION

        # Determine confidence level
        is_high_confidence = best_similarity >= CONFIG.HIGH_CONFIDENCE_THRESHOLD

        logger.info(f"[LABELED] '{label}' (score={best_similarity:.2f}): '{text[:80]}...'")

        return {
            "label": label,
            "score": float(best_similarity),
            "cluster_id": best_cluster.cluster_id,
            "theme": best_cluster.theme,
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
    cluster_count: int = 0
    ai_cluster_count: int = 0

state = AnalyzerState()


def initialize():
    """Initialize the analyzer with Community Notes data."""
    logger.info("=" * 60)
    logger.info("INITIALIZING LABELER")
    logger.info(f"Labels: '{LABEL_MISINFORMATION}', '{LABEL_AI_GENERATED}'")
    logger.info("=" * 60)

    # Validate config
    errors = CONFIG.validate()
    if errors:
        state.init_error = "; ".join(errors)
        logger.error(f"Configuration errors: {state.init_error}")
        return False

    try:
        # Initialize OpenAI client
        openai_client = openai.OpenAI(api_key=CONFIG.OPENAI_API_KEY)

        # Initialize embedding service
        embedding_service = EmbeddingService(CONFIG.OPENAI_API_KEY)

        # Initialize analyzer
        analyzer = MisinformationAnalyzer(embedding_service, openai_client)

        # Load Community Notes
        notes = analyzer.load_community_notes(CONFIG.NOTES_FILE)

        if notes:
            # Generate embeddings
            analyzer.generate_embeddings(notes)

            # Cluster notes
            clusters = analyzer.cluster_notes(notes)

            if clusters:
                state.cluster_count = len(clusters)
                state.ai_cluster_count = sum(1 for c in clusters if c.is_ai_related)

                logger.info("-" * 40)
                logger.info(f"Total clusters: {state.cluster_count}")
                logger.info(f"  - Misinformation: {state.cluster_count - state.ai_cluster_count}")
                logger.info(f"  - AI-generated: {state.ai_cluster_count}")
                logger.info("-" * 40)
            else:
                logger.warning("No clusters created from Community Notes")
        else:
            logger.warning("No Community Notes loaded - analyzer will return no labels")

        state.analyzer = analyzer
        state.initialized = True

        logger.info("=" * 60)
        logger.info("INITIALIZATION COMPLETE")
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
    title="Bluesky Content Labeler",
    description="Labels posts as 'misinformation' or 'ai-generated'",
    version="2.0.0",
    lifespan=lifespan
)


class PostPayload(BaseModel):
    """Request payload for post analysis."""
    text: str
    did: str  # Author DID


@app.post("/analyze")
async def analyze(payload: PostPayload):
    """
    Analyze a post for misinformation or AI-generated content.

    Returns:
        - {label: null} if no issues detected
        - {label: "misinformation", ...} for misinformation
        - {label: "ai-generated", ...} for AI/manipulated content
    """
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
        "clusters": state.cluster_count,
        "ai_clusters": state.ai_cluster_count,
        "labels": [LABEL_MISINFORMATION, LABEL_AI_GENERATED]
    }


@app.get("/stats")
async def stats():
    """Get analyzer statistics."""
    if not state.analyzer:
        return {"error": "Not initialized"}

    return {
        "total_clusters": len(state.analyzer.clusters),
        "misinformation_clusters": sum(1 for c in state.analyzer.clusters if not c.is_ai_related),
        "ai_generated_clusters": sum(1 for c in state.analyzer.clusters if c.is_ai_related),
        "keywords": len(state.analyzer.keyword_index),
        "labels": [LABEL_MISINFORMATION, LABEL_AI_GENERATED]
    }


@app.get("/labels")
async def get_labels():
    """Get the available label definitions."""
    return {
        "labelValues": [LABEL_MISINFORMATION, LABEL_AI_GENERATED],
        "labelValueDefinitions": [
            {
                "identifier": LABEL_MISINFORMATION,
                "severity": "inform",
                "blurs": "none",
                "defaultSetting": "warn",
                "locales": [{
                    "lang": "en",
                    "name": "Misinformation",
                    "description": "This post may contain misinformation based on Community Notes analysis."
                }]
            },
            {
                "identifier": LABEL_AI_GENERATED,
                "severity": "inform",
                "blurs": "none",
                "defaultSetting": "warn",
                "locales": [{
                    "lang": "en",
                    "name": "AI-Generated Content",
                    "description": "This post may contain AI-generated or manipulated media."
                }]
            }
        ]
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
