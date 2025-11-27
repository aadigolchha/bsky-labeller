"""
Production-Grade Bluesky Misinformation Labeler
================================================

Uses Community Notes data to identify and label misinformation on Bluesky.
Dynamically generates labels based on identified misinformation clusters.

This analyzer:
1. Loads Community Notes TSV data
2. Generates embeddings and clusters similar misinformation themes
3. Creates dynamic label definitions that can be registered with Bluesky
4. Analyzes incoming posts for similarity to known misinformation patterns
5. Uses LLM-based stance detection to avoid labeling counter-speech

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


@dataclass
class MisinfoCluster:
    """A cluster of semantically similar misinformation."""
    cluster_id: int
    notes: List[CommunityNote]
    centroid: np.ndarray
    theme: str  # Human-readable theme description
    label_id: str  # Bluesky-compatible label identifier (lowercase, alphanumeric + hyphens)
    label_name: str  # Display name for the label
    label_description: str  # Full description
    keywords: List[str]
    severity: str = "inform"  # inform, alert, or none
    blurs: str = "none"  # content, media, or none


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
# LABEL GENERATOR
# ============================================================================

class LabelGenerator:
    """Generates Bluesky label definitions from misinformation clusters."""
    
    def __init__(self, openai_client: openai.OpenAI):
        self.client = openai_client
    
    def sanitize_label_id(self, text: str) -> str:
        """Convert text to valid Bluesky label identifier."""
        # Lowercase, replace spaces with hyphens, remove non-alphanumeric
        sanitized = text.lower().strip()
        sanitized = re.sub(r'[^a-z0-9\s-]', '', sanitized)
        sanitized = re.sub(r'\s+', '-', sanitized)
        sanitized = re.sub(r'-+', '-', sanitized)
        sanitized = sanitized.strip('-')
        # Limit length
        return sanitized[:50] if sanitized else 'unknown'
    
    def generate_label_for_cluster(
        self, 
        cluster_id: int, 
        notes: List[CommunityNote]
    ) -> Tuple[str, str, str, str, List[str]]:
        """
        Use LLM to generate a meaningful label for a cluster of misinformation.
        
        Returns: (label_id, label_name, description, severity, keywords)
        """
        # Sample notes for analysis
        sample_summaries = [n.summary for n in notes[:5]]
        summaries_text = "\n".join(f"- {s}" for s in sample_summaries)
        
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
            # Fallback
            return (
                f"misinfo-{cluster_id}",
                f"Misinformation Pattern {cluster_id}",
                "Content flagged by Community Notes analysis.",
                "inform",
                []
            )
    
    def create_label_definition(self, cluster: MisinfoCluster) -> LabelDefinition:
        """Create a Bluesky-compatible label definition from a cluster."""
        return LabelDefinition(
            identifier=cluster.label_id,
            severity=cluster.severity,
            blurs=cluster.blurs,
            defaultSetting="warn",
            locales=[{
                "lang": "en",
                "name": cluster.label_name,
                "description": cluster.label_description
            }]
        )


# ============================================================================
# MISINFORMATION ANALYZER
# ============================================================================

class MisinformationAnalyzer:
    """Main analyzer that processes Community Notes and analyzes posts."""
    
    def __init__(self, embedding_service: EmbeddingService, openai_client: openai.OpenAI):
        self.embedding_service = embedding_service
        self.openai_client = openai_client
        self.label_generator = LabelGenerator(openai_client)
        self.clusters: List[MisinfoCluster] = []
        self.label_definitions: List[LabelDefinition] = []
        self.keyword_index: Dict[str, List[int]] = {}  # keyword -> cluster_ids
    
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
                
                note = CommunityNote(
                    note_id=row.get('noteId', str(i)),
                    tweet_id=row.get('tweetId', ''),
                    classification=classification,
                    summary=summary,
                    misleading_type=','.join(misleading_types) if misleading_types else 'general',
                    created_at=created_at
                )
                notes.append(note)
        
        logger.info(f"Loaded {len(notes)} valid Community Notes")
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
        """Cluster notes using DBSCAN and generate labels for each cluster."""
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
        cluster_notes: Dict[int, List[CommunityNote]] = defaultdict(list)
        for note, label in zip(notes_with_embeddings, clustering.labels_):
            if label >= 0:  # -1 is noise
                note.cluster_id = label
                cluster_notes[label].append(note)
        
        logger.info(f"Found {len(cluster_notes)} clusters (+ {sum(1 for l in clustering.labels_ if l == -1)} noise points)")
        
        # Build MisinfoCluster objects with generated labels
        clusters = []
        for cluster_id, notes_in_cluster in sorted(cluster_notes.items())[:CONFIG.MAX_CLUSTERS]:
            # Convert numpy int64 to Python int for JSON serialization
            cluster_id = int(cluster_id)
            
            # Compute centroid
            embeddings = np.vstack([n.embedding for n in notes_in_cluster])
            centroid = np.mean(embeddings, axis=0)
            
            # Generate label using LLM
            label_id, label_name, description, severity, keywords = \
                self.label_generator.generate_label_for_cluster(cluster_id, notes_in_cluster)
            
            cluster = MisinfoCluster(
                cluster_id=cluster_id,
                notes=notes_in_cluster,
                centroid=centroid,
                theme=label_name,
                label_id=label_id,
                label_name=label_name,
                label_description=description,
                keywords=keywords,
                severity=severity,
                blurs="none"
            )
            clusters.append(cluster)
            
            logger.info(f"Cluster {cluster_id}: '{label_name}' ({len(notes_in_cluster)} notes, {len(keywords)} keywords)")
        
        self.clusters = clusters
        self._build_keyword_index()
        self._generate_label_definitions()
        
        return clusters
    
    def _build_keyword_index(self):
        """Build an inverted index from keywords to cluster IDs for fast filtering."""
        self.keyword_index = defaultdict(list)
        for cluster in self.clusters:
            for keyword in cluster.keywords:
                kw_lower = keyword.lower()
                if cluster.cluster_id not in self.keyword_index[kw_lower]:
                    self.keyword_index[kw_lower].append(cluster.cluster_id)
        
        logger.info(f"Built keyword index with {len(self.keyword_index)} unique keywords")
    
    def _generate_label_definitions(self):
        """Generate Bluesky label definitions and save to file."""
        self.label_definitions = [
            self.label_generator.create_label_definition(c) for c in self.clusters
        ]
        
        # Save to JSON for use by publish-labels.ts
        output = {
            "labelValues": [d.identifier for d in self.label_definitions],
            "labelValueDefinitions": [d.to_dict() for d in self.label_definitions],
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "label_id": c.label_id,
                    "theme": c.theme,
                    "keywords": c.keywords,
                    "note_count": len(c.notes)
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
    
    def detect_stance(self, text: str, cluster: MisinfoCluster) -> str:
        """
        Determine if the text supports, refutes, or neutrally discusses the misinformation.
        
        This is critical to avoid labeling counter-speech and fact-checks.
        """
        prompt = f"""Analyze the stance of this social media post towards a known misinformation theme.

Misinformation theme: {cluster.theme}
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
            return 'neutral'  # Safe default
    
    def analyze_post(self, text: str, author_did: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a post for misinformation.
        
        Returns a dict with label info if misinformation is detected, None otherwise.
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
            return None
        
        # Stance detection - only label if post supports the misinformation
        stance = self.detect_stance(text, best_cluster)
        
        if stance != 'supports':
            logger.debug(f"Post matched cluster '{best_cluster.label_id}' but stance is '{stance}', not labeling")
            return None
        
        # Determine confidence level
        is_high_confidence = best_similarity >= CONFIG.HIGH_CONFIDENCE_THRESHOLD
        
        return {
            "label": best_cluster.label_id,
            "score": float(best_similarity),
            "cluster_id": best_cluster.cluster_id,
            "theme": best_cluster.theme,
            "severity": best_cluster.severity,
            "high_confidence": is_high_confidence,
            "text": text[:200]  # Truncate for logging
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
    logger.info("INITIALIZING MISINFORMATION ANALYZER")
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
            
            # Cluster and generate labels
            clusters = analyzer.cluster_notes(notes)
            
            if clusters:
                state.label_count = len(clusters)
                logger.info(f"Generated {len(clusters)} dynamic labels")
                
                # Log label summary
                logger.info("-" * 40)
                logger.info("REGISTERED LABELS:")
                for c in clusters[:10]:
                    logger.info(f"  - {c.label_id}: {c.label_name}")
                if len(clusters) > 10:
                    logger.info(f"  ... and {len(clusters) - 10} more")
                logger.info("-" * 40)
            else:
                logger.warning("No clusters created from Community Notes")
        else:
            logger.warning("No Community Notes loaded - analyzer will return no labels")
        
        state.analyzer = analyzer
        state.initialized = True
        
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
    version="1.0.0",
    lifespan=lifespan
)


class PostPayload(BaseModel):
    """Request payload for post analysis."""
    text: str
    did: str  # Author DID


@app.post("/analyze")
async def analyze(payload: PostPayload):
    """
    Analyze a post for misinformation.
    
    Returns:
        - {label: null} if no misinformation detected
        - {label: "label-id", score: 0.XX, ...} if misinformation detected
    """
    if not state.initialized or state.analyzer is None:
        # Not initialized - return no label (don't fail the request)
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
                "note_count": len(c.notes)
            }
            for c in state.analyzer.clusters
        ]
    }


@app.get("/labels")
async def get_labels():
    """
    Get the generated label definitions.
    
    This can be used to update publish-labels.ts with the dynamically generated labels.
    """
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