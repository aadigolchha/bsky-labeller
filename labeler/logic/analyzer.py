"""
Production-grade Bluesky Misinformation Analyzer
Uses Community Notes data for real-time misinformation detection via embeddings and clustering.
Designed to work with the TypeScript labeler infrastructure.
"""

import os
import json
import hashlib
import logging
import pickle
import csv
from datetime import datetime
from typing import List, Dict, Optional, Tuple
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    'EMBEDDING_MODEL': 'text-embedding-3-small',
    'EMBEDDING_BATCH_SIZE': 100,
    'CLUSTERING_EPS': 0.3,
    'CLUSTERING_MIN_SAMPLES': 2,
    'CONFIDENCE_THRESHOLD': 0.75,
    'STANCE_DETECTION_THRESHOLD': 0.45,
    'MAX_NOTES': int(os.getenv('MAX_NOTES', '1000')),
    'CACHE_DIR': '/data/cache' if os.getenv('NODE_ENV') == 'production' else './cache',
    'NOTES_FILE': os.getenv('NOTES_FILE', 'notes-00000.tsv'),
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CommunityNote:
    """Represents a Community Note from X/Twitter"""
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
    """Represents a cluster of similar misinformation"""
    cluster_id: int
    notes: List[CommunityNote]
    centroid: np.ndarray
    theme: str
    keywords: List[str]
    confidence: float


# ============================================================================
# EMBEDDING SERVICE
# ============================================================================

class EmbeddingService:
    """Service for generating embeddings using OpenAI"""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = CONFIG['EMBEDDING_MODEL']
        self.cache_dir = Path(CONFIG['CACHE_DIR'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load embedding cache from disk"""
        cache_file = self.cache_dir / 'embeddings.pkl'
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                logger.error(f"Failed to load embedding cache: {e}")
    
    def _save_cache(self):
        """Save embedding cache to disk"""
        cache_file = self.cache_dir / 'embeddings.pkl'
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a single text"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            embedding = np.array(response.data[0].embedding)
            self.embedding_cache[text_hash] = embedding
            self._save_cache()
            return embedding
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return None
    
    def get_embeddings_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Get embeddings for multiple texts in batches"""
        embeddings = []
        batch_size = CONFIG['EMBEDDING_BATCH_SIZE']
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings: List[Optional[np.ndarray]] = []
            uncached_texts = []
            uncached_indices = []
            
            # Check cache first
            for j, text in enumerate(batch):
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash in self.embedding_cache:
                    batch_embeddings.append(self.embedding_cache[text_hash])
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(j)
                    batch_embeddings.append(None)
            
            # Get embeddings for uncached texts
            if uncached_texts:
                try:
                    response = self.client.embeddings.create(
                        input=uncached_texts,
                        model=self.model
                    )
                    
                    for idx, embedding_data in zip(uncached_indices, response.data):
                        embedding = np.array(embedding_data.embedding)
                        batch_embeddings[idx] = embedding
                        text_hash = hashlib.md5(batch[idx].encode()).hexdigest()
                        self.embedding_cache[text_hash] = embedding
                    
                    self._save_cache()
                except Exception as e:
                    logger.error(f"Failed to get batch embeddings: {e}")
            
            embeddings.extend(batch_embeddings)
            logger.info(f"Processed batch {i//batch_size + 1}, total embeddings: {len(embeddings)}")
        
        return embeddings


# ============================================================================
# MISINFORMATION ANALYZER
# ============================================================================

class MisinformationAnalyzer:
    """Analyzes content for misinformation using Community Notes data"""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.clusters: List[MisinfoCluster] = []
        self.keyword_filter: set = set()
        self.openai_client = openai.OpenAI(api_key=CONFIG['OPENAI_API_KEY'])
    
    def load_community_notes(self, tsv_path: str, max_notes: Optional[int] = None) -> List[CommunityNote]:
        """Load Community Notes from TSV file"""
        notes = []
        
        if not Path(tsv_path).exists():
            logger.warning(f"Community Notes file not found: {tsv_path}")
            return notes
        
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            for i, row in enumerate(reader):
                if max_notes and i >= max_notes:
                    break
                
                # Filter for misleading notes with summaries
                if (row.get('classification') == 'MISINFORMED_OR_POTENTIALLY_MISLEADING' and
                    row.get('summary') and len(row.get('summary', '')) > 10):
                    
                    # Determine misleading type
                    misleading_types = []
                    if row.get('misleadingFactualError') == '1':
                        misleading_types.append('factual_error')
                    if row.get('misleadingManipulatedMedia') == '1':
                        misleading_types.append('manipulated_media')
                    if row.get('misleadingMissingImportantContext') == '1':
                        misleading_types.append('missing_context')
                    if row.get('misleadingUnverifiedClaimAsFact') == '1':
                        misleading_types.append('unverified_claim')
                    
                    try:
                        created_at = datetime.fromtimestamp(int(row['createdAtMillis']) / 1000)
                    except (ValueError, KeyError):
                        created_at = datetime.now()
                    
                    note = CommunityNote(
                        note_id=row.get('noteId', ''),
                        tweet_id=row.get('tweetId', ''),
                        classification=row.get('classification', ''),
                        summary=row.get('summary', ''),
                        misleading_type=','.join(misleading_types) if misleading_types else 'other',
                        created_at=created_at
                    )
                    notes.append(note)
        
        logger.info(f"Loaded {len(notes)} Community Notes")
        return notes
    
    def generate_embeddings(self, notes: List[CommunityNote]):
        """Generate embeddings for Community Notes"""
        texts = [note.summary for note in notes]
        embeddings = self.embedding_service.get_embeddings_batch(texts)
        
        for note, embedding in zip(notes, embeddings):
            if embedding is not None:
                note.embedding = embedding
        
        valid_count = sum(1 for n in notes if n.embedding is not None)
        logger.info(f"Generated embeddings for {valid_count}/{len(notes)} notes")
    
    def cluster_notes(self, notes: List[CommunityNote]) -> List[MisinfoCluster]:
        """Cluster similar Community Notes using DBSCAN"""
        notes_with_embeddings = [n for n in notes if n.embedding is not None]
        
        if not notes_with_embeddings:
            logger.warning("No notes with embeddings to cluster")
            return []
        
        X = np.array([n.embedding for n in notes_with_embeddings])
        
        clustering = DBSCAN(
            eps=CONFIG['CLUSTERING_EPS'],
            min_samples=CONFIG['CLUSTERING_MIN_SAMPLES'],
            metric='cosine'
        ).fit(X)
        
        clusters_dict = defaultdict(list)
        for note, label in zip(notes_with_embeddings, clustering.labels_):
            if label != -1:  # Ignore noise
                note.cluster_id = label
                clusters_dict[label].append(note)
        
        misinfo_clusters = []
        for cluster_id, cluster_notes in clusters_dict.items():
            embeddings = np.array([n.embedding for n in cluster_notes])
            centroid = np.mean(embeddings, axis=0)
            
            # Extract theme and keywords
            theme, keywords = self._extract_cluster_theme(cluster_notes)
            
            cluster = MisinfoCluster(
                cluster_id=int(cluster_id),
                notes=cluster_notes,
                centroid=centroid,
                theme=theme,
                keywords=keywords,
                confidence=0.9
            )
            misinfo_clusters.append(cluster)
        
        logger.info(f"Created {len(misinfo_clusters)} misinformation clusters")
        self.clusters = misinfo_clusters
        return misinfo_clusters
    
    def _extract_cluster_theme(self, notes: List[CommunityNote]) -> Tuple[str, List[str]]:
        """Extract theme and keywords from a cluster using GPT"""
        sample_notes = notes[:5]
        summaries = '\n'.join([f"- {n.summary}" for n in sample_notes])
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing misinformation patterns. Extract the common theme and keywords from these Community Notes summaries."
                    },
                    {
                        "role": "user",
                        "content": f"""Analyze these Community Notes summaries and provide:
1. A brief theme description (max 100 characters)
2. 5-10 relevant keywords

Summaries:
{summaries}

Respond in JSON format:
{{"theme": "...", "keywords": ["...", "..."]}}"""
                    }
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('theme', 'Unknown theme'), result.get('keywords', [])
        except Exception as e:
            logger.error(f"Failed to extract cluster theme: {e}")
            return "Unknown theme", []
    
    def build_keyword_filter(self):
        """Build keyword filter from cluster keywords for pre-filtering"""
        self.keyword_filter = set()
        for cluster in self.clusters:
            for keyword in cluster.keywords:
                self.keyword_filter.add(keyword.lower())
        logger.info(f"Built keyword filter with {len(self.keyword_filter)} keywords")
    
    def should_analyze_post(self, text: str) -> bool:
        """Quick keyword-based check to filter out irrelevant posts"""
        if not text or not self.keyword_filter:
            return True
        
        text_lower = text.lower()
        for keyword in self.keyword_filter:
            if keyword in text_lower:
                return True
        return False
    
    def analyze_post(self, text: str, did: str) -> Optional[Tuple[str, float, int]]:
        """
        Analyze a post for misinformation.
        Returns: (label, confidence, cluster_id) or None
        """
        if not text or not self.clusters:
            return None
        
        # Quick keyword filter
        if not self.should_analyze_post(text):
            return None
        
        # Generate embedding
        post_embedding = self.embedding_service.get_embedding(text)
        if post_embedding is None:
            return None
        
        # Compare with cluster centroids
        best_match = None
        best_similarity = 0.0
        
        for cluster in self.clusters:
            similarity = cosine_similarity(
                post_embedding.reshape(1, -1),
                cluster.centroid.reshape(1, -1)
            )[0, 0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cluster
        
        if best_similarity < CONFIG['STANCE_DETECTION_THRESHOLD']:
            return None
        
        # Stance detection
        stance = self._detect_stance(text, best_match)
        
        if stance == 'supports':
            if best_similarity >= CONFIG['CONFIDENCE_THRESHOLD']:
                return 'misinformation', best_similarity, best_match.cluster_id
            else:
                return 'potential-misinformation', best_similarity, best_match.cluster_id
        
        return None
    
    def _detect_stance(self, text: str, cluster: MisinfoCluster) -> str:
        """Detect stance of text towards misinformation"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at stance detection. Determine if the given text supports, refutes, or is neutral towards a claim."
                    },
                    {
                        "role": "user",
                        "content": f"""Misinformation theme: {cluster.theme}
Keywords: {', '.join(cluster.keywords)}

Text to analyze:
"{text}"

Does this text support, refute, or discuss neutrally the misinformation?
Respond with only one word: "supports", "refutes", or "neutral"."""
                    }
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            stance = response.choices[0].message.content.strip().lower()
            if stance not in ['supports', 'refutes', 'neutral']:
                return 'neutral'
            return stance
        except Exception as e:
            logger.error(f"Stance detection failed: {e}")
            return 'neutral'


# ============================================================================
# GLOBAL STATE & INITIALIZATION
# ============================================================================

class AnalyzerState:
    """Global state for the analyzer"""
    analyzer: Optional[MisinformationAnalyzer] = None
    initialized: bool = False
    init_error: Optional[str] = None

state = AnalyzerState()


def initialize_analyzer():
    """Initialize the misinformation analyzer with Community Notes data"""
    logger.info("Initializing Misinformation Analyzer...")
    
    if not CONFIG['OPENAI_API_KEY']:
        state.init_error = "OPENAI_API_KEY not configured"
        logger.error(state.init_error)
        return False
    
    try:
        # Initialize services
        embedding_service = EmbeddingService(CONFIG['OPENAI_API_KEY'])
        analyzer = MisinformationAnalyzer(embedding_service)
        
        # Load Community Notes
        notes_file = CONFIG['NOTES_FILE']
        notes = analyzer.load_community_notes(notes_file, CONFIG['MAX_NOTES'])
        
        if notes:
            # Generate embeddings
            logger.info("Generating embeddings for Community Notes...")
            analyzer.generate_embeddings(notes)
            
            # Cluster notes
            logger.info("Clustering notes to identify misinformation themes...")
            clusters = analyzer.cluster_notes(notes)
            
            if clusters:
                analyzer.build_keyword_filter()
                
                # Log cluster info
                for cluster in clusters[:5]:
                    logger.info(f"Cluster: {cluster.theme}")
                    logger.info(f"  Keywords: {', '.join(cluster.keywords[:5])}")
                    logger.info(f"  Notes: {len(cluster.notes)}")
        else:
            logger.warning("No Community Notes loaded - running in fallback mode")
        
        state.analyzer = analyzer
        state.initialized = True
        logger.info("Misinformation Analyzer initialized successfully!")
        return True
        
    except Exception as e:
        state.init_error = str(e)
        logger.error(f"Failed to initialize analyzer: {e}")
        return False


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize on startup"""
    initialize_analyzer()
    yield
    logger.info("Shutting down analyzer...")

app = FastAPI(lifespan=lifespan)


class PostPayload(BaseModel):
    text: str
    did: str


@app.post("/analyze")
async def analyze(payload: PostPayload):
    """Analyze a post for misinformation"""
    
    # Fallback to simple keyword matching if not initialized
    if not state.initialized or state.analyzer is None:
        text_lower = payload.text.lower()
        
        # Fallback keyword matching
        if "election news" in text_lower:
            return {"label": "election", "score": 1.0, "text": text_lower}
        if "vaccination" in text_lower or "vaccine" in text_lower:
            return {"label": "vaccine", "score": 1.0, "text": text_lower}
        
        return {"label": None}
    
    # Use sophisticated analysis
    result = state.analyzer.analyze_post(payload.text, payload.did)
    
    if result:
        label, confidence, cluster_id = result
        
        # Get theme for logging
        theme = None
        for cluster in state.analyzer.clusters:
            if cluster.cluster_id == cluster_id:
                theme = cluster.theme
                break
        
        return {
            "label": label,
            "score": confidence,
            "cluster_id": cluster_id,
            "theme": theme,
            "text": payload.text
        }
    
    return {"label": None}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "initialized": state.initialized,
        "clusters": len(state.analyzer.clusters) if state.analyzer else 0,
        "keywords": len(state.analyzer.keyword_filter) if state.analyzer else 0,
        "error": state.init_error
    }


@app.get("/stats")
async def stats():
    """Get analyzer statistics"""
    if not state.analyzer:
        return {"error": "Not initialized"}
    
    return {
        "clusters": len(state.analyzer.clusters),
        "keywords": len(state.analyzer.keyword_filter),
        "themes": [
            {
                "id": c.cluster_id,
                "theme": c.theme,
                "keywords": c.keywords[:5],
                "notes": len(c.notes)
            }
            for c in state.analyzer.clusters[:10]
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)