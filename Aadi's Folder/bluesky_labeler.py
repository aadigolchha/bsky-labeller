#!/usr/bin/env python3
"""
Production-grade Bluesky Misinformation Labeling Service
Uses Community Notes data to identify and label misinformation on Bluesky
"""

import os
import sys
import json
import time
import hashlib
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import csv
import pickle

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import openai
from dotenv import load_dotenv
import requests
from atproto import Client, models, FirehoseSubscribeReposClient, parse_subscribe_repos_message, CAR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bluesky_labeler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    'BLUESKY_HANDLE': os.getenv('BLUESKY_HANDLE', 'misinfolabeler.bsky.social'),
    'BLUESKY_PASSWORD': os.getenv('BLUESKY_PASSWORD'),
    'EMBEDDING_MODEL': 'text-embedding-3-small',
    'EMBEDDING_BATCH_SIZE': 100,
    'CLUSTERING_EPS': 0.3,
    'CLUSTERING_MIN_SAMPLES': 2,
    'CONFIDENCE_THRESHOLD': 0.75,
    'STANCE_DETECTION_THRESHOLD': 0.45,  # Lowered from 0.7 to capture more relevant posts (social media posts typically have 0.3-0.6 similarity to cluster centroids)
    'MAX_NOTES_FOR_TESTING': 1000,  # Set to None for production, or change number for testing (increased from 100)
    'CACHE_DIR': './cache',
    'DB_PATH': './labeler.db',
    'FIREHOSE_BUFFER_SIZE': 10000,
    'BATCH_PROCESSING_INTERVAL': 30,  # seconds
    'LABEL_SERVICE_DID': None,  # Will be set after registration
}

# Bluesky label values
LABEL_VALUES = {
    'misinformation': {
        'identifier': 'misinformation',
        'severity': 'inform',
        'defaultSetting': 'warn',
        'blurs': 'content',
        'locales': [
            {
                'lang': 'en',
                'name': 'Misinformation',
                'description': 'Content identified as containing misinformation based on Community Notes data'
            }
        ]
    },
    'potential-misinformation': {
        'identifier': 'potential-misinformation',
        'severity': 'inform',
        'defaultSetting': 'none',
        'blurs': 'none',
        'locales': [
            {
                'lang': 'en',
                'name': 'Potential Misinformation',
                'description': 'Content that may contain misinformation (lower confidence)'
            }
        ]
    }
}


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


@dataclass
class BlueskyPost:
    """Represents a Bluesky post to be analyzed"""
    uri: str
    cid: str
    author_did: str
    text: str
    created_at: datetime
    embedding: Optional[np.ndarray] = None
    labels: List[str] = None


class DatabaseManager:
    """Manages SQLite database for persistent storage"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Community notes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS community_notes (
                note_id TEXT PRIMARY KEY,
                tweet_id TEXT,
                classification TEXT,
                summary TEXT,
                misleading_type TEXT,
                created_at TIMESTAMP,
                embedding BLOB,
                cluster_id INTEGER
            )
        ''')
        
        # Misinfo clusters table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS misinfo_clusters (
                cluster_id INTEGER PRIMARY KEY,
                theme TEXT,
                keywords TEXT,
                centroid BLOB,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Labeled posts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS labeled_posts (
                uri TEXT PRIMARY KEY,
                cid TEXT,
                author_did TEXT,
                text TEXT,
                label TEXT,
                confidence REAL,
                cluster_id INTEGER,
                created_at TIMESTAMP,
                labeled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (cluster_id) REFERENCES misinfo_clusters(cluster_id)
            )
        ''')
        
        # Processing queue table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uri TEXT UNIQUE,
                cid TEXT,
                author_did TEXT,
                text TEXT,
                created_at TIMESTAMP,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT 0
            )
        ''')
        
        self.conn.commit()
    
    def save_community_note(self, note: CommunityNote):
        """Save a community note to database"""
        cursor = self.conn.cursor()
        embedding_blob = pickle.dumps(note.embedding) if note.embedding is not None else None
        
        cursor.execute('''
            INSERT OR REPLACE INTO community_notes
            (note_id, tweet_id, classification, summary, misleading_type, created_at, embedding, cluster_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            note.note_id, note.tweet_id, note.classification,
            note.summary, note.misleading_type, note.created_at,
            embedding_blob, note.cluster_id
        ))
        self.conn.commit()
    
    def save_cluster(self, cluster: MisinfoCluster):
        """Save a misinfo cluster to database"""
        cursor = self.conn.cursor()
        centroid_blob = pickle.dumps(cluster.centroid)
        keywords_json = json.dumps(cluster.keywords)
        
        cursor.execute('''
            INSERT OR REPLACE INTO misinfo_clusters
            (cluster_id, theme, keywords, centroid, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            cluster.cluster_id, cluster.theme, keywords_json,
            centroid_blob, cluster.confidence
        ))
        self.conn.commit()
    
    def get_clusters(self) -> List[MisinfoCluster]:
        """Retrieve all misinfo clusters from database"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM misinfo_clusters')
        clusters = []
        
        for row in cursor.fetchall():
            cluster_id, theme, keywords_json, centroid_blob, confidence, _ = row
            centroid = pickle.loads(centroid_blob)
            keywords = json.loads(keywords_json)
            
            # Get associated notes
            cursor.execute(
                'SELECT * FROM community_notes WHERE cluster_id = ?',
                (cluster_id,)
            )
            notes = []
            for note_row in cursor.fetchall():
                note = CommunityNote(
                    note_id=note_row[0],
                    tweet_id=note_row[1],
                    classification=note_row[2],
                    summary=note_row[3],
                    misleading_type=note_row[4],
                    created_at=note_row[5],
                    embedding=pickle.loads(note_row[6]) if note_row[6] else None,
                    cluster_id=note_row[7]
                )
                notes.append(note)
            
            cluster = MisinfoCluster(
                cluster_id=cluster_id,
                notes=notes,
                centroid=centroid,
                theme=theme,
                keywords=keywords,
                confidence=confidence
            )
            clusters.append(cluster)
        
        return clusters
    
    def add_to_queue(self, post: BlueskyPost):
        """Add a post to the processing queue"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO processing_queue
            (uri, cid, author_did, text, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (post.uri, post.cid, post.author_did, post.text, post.created_at))
        self.conn.commit()
    
    def get_unprocessed_posts(self, limit: int = 100) -> List[BlueskyPost]:
        """Get unprocessed posts from queue"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT uri, cid, author_did, text, created_at
            FROM processing_queue
            WHERE processed = 0
            LIMIT ?
        ''', (limit,))
        
        posts = []
        for row in cursor.fetchall():
            post = BlueskyPost(
                uri=row[0],
                cid=row[1],
                author_did=row[2],
                text=row[3],
                created_at=row[4]
            )
            posts.append(post)
        
        return posts
    
    def mark_processed(self, uris: List[str]):
        """Mark posts as processed"""
        cursor = self.conn.cursor()
        cursor.executemany(
            'UPDATE processing_queue SET processed = 1 WHERE uri = ?',
            [(uri,) for uri in uris]
        )
        self.conn.commit()
    
    def save_labeled_post(self, post: BlueskyPost, label: str, confidence: float, cluster_id: int):
        """Save a labeled post"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO labeled_posts
            (uri, cid, author_did, text, label, confidence, cluster_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            post.uri, post.cid, post.author_did, post.text,
            label, confidence, cluster_id, post.created_at
        ))
        self.conn.commit()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class EmbeddingService:
    """Service for generating embeddings using OpenAI"""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = CONFIG['EMBEDDING_MODEL']
        self.cache_dir = CONFIG['CACHE_DIR']
        os.makedirs(self.cache_dir, exist_ok=True)
        self.embedding_cache = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load embedding cache from disk"""
        cache_file = os.path.join(self.cache_dir, 'embeddings.pkl')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                logger.error(f"Failed to load embedding cache: {e}")
    
    def _save_cache(self):
        """Save embedding cache to disk"""
        cache_file = os.path.join(self.cache_dir, 'embeddings.pkl')
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text"""
        # Check cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            embedding = np.array(response.data[0].embedding)
            
            # Cache the embedding
            self.embedding_cache[text_hash] = embedding
            self._save_cache()
            
            return embedding
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return None
    
    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts in batches"""
        embeddings = []
        batch_size = CONFIG['EMBEDDING_BATCH_SIZE']
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Check cache first
            batch_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
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
                        
                        # Cache the embedding
                        text_hash = hashlib.md5(batch[idx].encode()).hexdigest()
                        self.embedding_cache[text_hash] = embedding
                    
                    self._save_cache()
                    
                except Exception as e:
                    logger.error(f"Failed to get batch embeddings: {e}")
            
            embeddings.extend(batch_embeddings)
        
        return embeddings


class MisinformationAnalyzer:
    """Analyzes content for misinformation using Community Notes data"""
    
    def __init__(self, embedding_service: EmbeddingService, db_manager: DatabaseManager):
        self.embedding_service = embedding_service
        self.db_manager = db_manager
        self.clusters = []
        self.openai_client = openai.OpenAI(api_key=CONFIG['OPENAI_API_KEY'])
    
    def load_community_notes(self, tsv_path: str, max_notes: Optional[int] = None) -> List[CommunityNote]:
        """Load Community Notes from TSV file"""
        notes = []
        
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            for i, row in enumerate(reader):
                if max_notes and i >= max_notes:
                    break
                
                # Filter for misleading notes with summaries
                if (row['classification'] == 'MISINFORMED_OR_POTENTIALLY_MISLEADING' and
                    row['summary'] and len(row['summary']) > 10):
                    
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
                    
                    note = CommunityNote(
                        note_id=row['noteId'],
                        tweet_id=row['tweetId'],
                        classification=row['classification'],
                        summary=row['summary'],
                        misleading_type=','.join(misleading_types) if misleading_types else 'other',
                        created_at=datetime.fromtimestamp(int(row['createdAtMillis']) / 1000)
                    )
                    notes.append(note)
                    
                    # Save to database
                    self.db_manager.save_community_note(note)
        
        logger.info(f"Loaded {len(notes)} Community Notes")
        return notes
    
    def generate_embeddings(self, notes: List[CommunityNote]):
        """Generate embeddings for Community Notes"""
        texts = [note.summary for note in notes]
        embeddings = self.embedding_service.get_embeddings_batch(texts)
        
        for note, embedding in zip(notes, embeddings):
            if embedding is not None:
                note.embedding = embedding
                self.db_manager.save_community_note(note)
        
        logger.info(f"Generated embeddings for {len(notes)} notes")
    
    def cluster_notes(self, notes: List[CommunityNote]) -> List[MisinfoCluster]:
        """Cluster similar Community Notes using DBSCAN"""
        # Filter notes with embeddings
        notes_with_embeddings = [n for n in notes if n.embedding is not None]
        
        if not notes_with_embeddings:
            logger.warning("No notes with embeddings to cluster")
            return []
        
        # Create embedding matrix
        X = np.array([n.embedding for n in notes_with_embeddings])
        
        # Perform clustering
        clustering = DBSCAN(
            eps=CONFIG['CLUSTERING_EPS'],
            min_samples=CONFIG['CLUSTERING_MIN_SAMPLES'],
            metric='cosine'
        ).fit(X)
        
        # Group notes by cluster
        clusters = defaultdict(list)
        for note, label in zip(notes_with_embeddings, clustering.labels_):
            if label != -1:  # Ignore noise points
                note.cluster_id = label
                clusters[label].append(note)
                self.db_manager.save_community_note(note)
        
        # Create MisinfoCluster objects
        misinfo_clusters = []
        for cluster_id, cluster_notes in clusters.items():
            # Calculate centroid
            embeddings = np.array([n.embedding for n in cluster_notes])
            centroid = np.mean(embeddings, axis=0)
            
            # Extract theme and keywords using GPT
            theme, keywords = self._extract_cluster_theme(cluster_notes)
            
            cluster = MisinfoCluster(
                cluster_id=int(cluster_id),  # Convert numpy int to Python int
                notes=cluster_notes,
                centroid=centroid,
                theme=theme,
                keywords=keywords,
                confidence=0.9  # Base confidence for clustered notes
            )
            
            misinfo_clusters.append(cluster)
            self.db_manager.save_cluster(cluster)
        
        logger.info(f"Created {len(misinfo_clusters)} misinfo clusters")
        self.clusters = misinfo_clusters
        return misinfo_clusters
    
    def _extract_cluster_theme(self, notes: List[CommunityNote]) -> Tuple[str, List[str]]:
        """Extract theme and keywords from a cluster of notes using GPT"""
        # Sample up to 5 notes for theme extraction
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
    
    def analyze_post(self, post: BlueskyPost) -> Optional[Tuple[str, float, int]]:
        """
        Analyze a Bluesky post for misinformation
        Returns: (label, confidence, cluster_id) or None
        """
        if not post.text or not self.clusters:
            return None
        
        # Generate embedding for post
        post_embedding = self.embedding_service.get_embedding(post.text)
        if post_embedding is None:
            return None
        
        post.embedding = post_embedding
        
        # Compare with cluster centroids
        best_match = None
        best_similarity = 0
        
        for cluster in self.clusters:
            similarity = cosine_similarity(
                post_embedding.reshape(1, -1),
                cluster.centroid.reshape(1, -1)
            )[0, 0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cluster
        
        # Check if similarity exceeds threshold
        if best_similarity < CONFIG['STANCE_DETECTION_THRESHOLD']:
            return None
        
        # Perform stance detection to determine if post supports the misinfo
        stance = self._detect_stance(post.text, best_match)
        
        if stance == 'supports':
            if best_similarity >= CONFIG['CONFIDENCE_THRESHOLD']:
                return 'misinformation', best_similarity, best_match.cluster_id
            else:
                return 'potential-misinformation', best_similarity, best_match.cluster_id
        
        return None
    
    def _detect_stance(self, text: str, cluster: MisinfoCluster) -> str:
        """
        Detect stance of text towards misinformation
        Returns: 'supports', 'refutes', or 'neutral'
        """
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


class BlueskyLabelService:
    """Bluesky label service implementation"""

    def __init__(self, analyzer: MisinformationAnalyzer, db_manager: DatabaseManager):
        self.analyzer = analyzer
        self.db_manager = db_manager
        self.client = None
        self.service_did = None
        self.firehose_thread = None
        self.processing_thread = None
        self.running = False
        self.summary_file = "labeling_summary.txt"
        self.keyword_filter = set()  # Will be populated from cluster keywords
        self.filtered_count = 0  # Track how many posts we filter out
        self.processed_count = 0  # Track total posts seen

    def build_keyword_filter(self, clusters: List[MisinfoCluster]):
        """Build keyword filter from cluster keywords for pre-filtering"""
        self.keyword_filter = set()
        for cluster in clusters:
            for keyword in cluster.keywords:
                # Add lowercase keywords for case-insensitive matching
                self.keyword_filter.add(keyword.lower())
        logger.info(f"Built keyword filter with {len(self.keyword_filter)} keywords: {sorted(list(self.keyword_filter))[:20]}")

    def should_analyze_post(self, text: str) -> bool:
        """Quick keyword-based check to filter out irrelevant posts"""
        if not text or not self.keyword_filter:
            return True  # If no filter, analyze everything

        self.processed_count += 1
        text_lower = text.lower()
        # Check if post contains any cluster keywords
        for keyword in self.keyword_filter:
            if keyword in text_lower:
                return True

        self.filtered_count += 1
        return False

    def write_themes_summary(self, clusters: List[MisinfoCluster]):
        """Write community notes themes summary to file"""
        try:
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("BLUESKY MISINFORMATION LABELING SERVICE - SUMMARY\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Misinformation Themes Identified: {len(clusters)}\n\n")

                f.write("=" * 80 + "\n")
                f.write("COMMUNITY NOTES THEMES\n")
                f.write("=" * 80 + "\n\n")

                for i, cluster in enumerate(clusters, 1):
                    f.write(f"\n--- Theme #{i} (Cluster ID: {cluster.cluster_id}) ---\n")
                    f.write(f"Theme: {cluster.theme}\n")
                    f.write(f"Keywords: {', '.join(cluster.keywords)}\n")
                    f.write(f"Number of Community Notes: {len(cluster.notes)}\n")
                    f.write(f"Confidence: {cluster.confidence:.2f}\n\n")

                    f.write("Sample Community Notes:\n")
                    for j, note in enumerate(cluster.notes[:3], 1):  # Show up to 3 notes per cluster
                        f.write(f"  {j}. {note.summary}\n")
                    f.write("\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("LABELED BLUESKY POSTS\n")
                f.write("=" * 80 + "\n")
                f.write("(Posts will be appended below as they are labeled)\n\n")

            logger.info(f"Themes summary written to {self.summary_file}")
        except Exception as e:
            logger.error(f"Failed to write themes summary: {e}")

    def append_labeled_post(self, post: BlueskyPost, label: str, confidence: float, cluster: MisinfoCluster):
        """Append labeled post information to summary file"""
        try:
            with open(self.summary_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "-" * 80 + "\n")
                f.write(f"LABELED POST - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("-" * 80 + "\n\n")
                f.write(f"Post URI: {post.uri}\n")
                f.write(f"Author DID: {post.author_did}\n")
                f.write(f"Posted: {post.created_at}\n\n")
                f.write(f"Content:\n{post.text}\n\n")
                f.write(f"Label Applied: {label}\n")
                f.write(f"Confidence: {confidence:.2f}\n\n")
                f.write(f"Matched Community Note Theme:\n")
                f.write(f"  Theme: {cluster.theme}\n")
                f.write(f"  Keywords: {', '.join(cluster.keywords[:10])}\n")
                f.write(f"  Cluster ID: {cluster.cluster_id}\n\n")
                f.write(f"Sample Community Notes from this theme:\n")
                for i, note in enumerate(cluster.notes[:2], 1):
                    f.write(f"  {i}. {note.summary}\n")
                f.write("\n")

            logger.info(f"Labeled post appended to {self.summary_file}")
        except Exception as e:
            logger.error(f"Failed to append labeled post: {e}")

    def authenticate(self):
        """Authenticate with Bluesky"""
        self.client = Client()
        self.client.login(
            CONFIG['BLUESKY_HANDLE'],
            CONFIG['BLUESKY_PASSWORD']
        )
        logger.info(f"Authenticated as {CONFIG['BLUESKY_HANDLE']}")

        # Get service DID
        profile = self.client.get_profile(CONFIG['BLUESKY_HANDLE'])
        self.service_did = profile.did
        CONFIG['LABEL_SERVICE_DID'] = self.service_did
    
    def register_labeler(self):
        """Register as a label service on Bluesky"""
        try:
            # Create labeler declaration
            labeler_config = {
                "$type": "app.bsky.labeler.service",
                "policies": {
                    "labelValues": list(LABEL_VALUES.keys())
                },
                "createdAt": datetime.now().isoformat() + "Z"
            }
            
            # Post labeler declaration
            data = models.com.atproto.repo.put_record.Data(
                repo=self.service_did,
                collection="app.bsky.labeler.service",
                rkey="self",
                record=labeler_config
            )
            self.client.com.atproto.repo.put_record(data=data)
            
            logger.info("Registered as Bluesky label service")
            
        except Exception as e:
            logger.error(f"Failed to register labeler: {e}")
    
    def subscribe_to_firehose(self):
        """Subscribe to Bluesky firehose for real-time posts"""
        def on_message_handler(message):
            if not self.running:
                return

            try:
                commit = parse_subscribe_repos_message(message)
                if not hasattr(commit, 'ops'):
                    return

                # Parse CAR blocks to access actual record data
                car = CAR.from_bytes(commit.blocks)

                for op in commit.ops:
                    # Filter for post creates
                    if op.action == 'create' and 'app.bsky.feed.post' in op.path:
                        try:
                            # Parse the record from CAR blocks
                            if hasattr(op, 'cid') and op.cid:
                                try:
                                    # Get the actual record data from CAR blocks
                                    record_raw_data = car.blocks.get(op.cid)

                                    if record_raw_data:
                                        # Deserialize the record
                                        record = models.get_or_create(record_raw_data, strict=False)

                                        # Extract text if this is a post
                                        text_content = ''
                                        if models.is_record_type(record, models.ids.AppBskyFeedPost):
                                            text_content = record.text if hasattr(record, 'text') else ''

                                        if text_content:  # Only queue posts with text
                                            post = BlueskyPost(
                                                uri=f"at://{commit.repo}/{op.path}",
                                                cid=str(op.cid),
                                                author_did=commit.repo,
                                                text=text_content,
                                                created_at=datetime.now()
                                            )

                                            # Add to processing queue
                                            self.db_manager.add_to_queue(post)

                                except Exception as e:
                                    logger.error(f"Error extracting text from CAR blocks: {e}")

                        except Exception as e:
                            logger.error(f"Error processing firehose event: {e}")

            except Exception as e:
                logger.error(f"Error parsing firehose message: {e}")

        def start_firehose():
            while self.running:
                try:
                    logger.info("Connecting to firehose...")
                    client = FirehoseSubscribeReposClient()
                    client.start(on_message_handler)

                except Exception as e:
                    logger.error(f"Firehose connection error: {e}")
                    if self.running:
                        time.sleep(5)  # Wait before reconnecting

        self.firehose_thread = threading.Thread(target=start_firehose, daemon=True)
        self.firehose_thread.start()
        logger.info("Started firehose subscription")
    
    def start_batch_processor(self):
        """Start batch processing thread"""
        def process_batches():
            while self.running:
                try:
                    # Get unprocessed posts
                    posts = self.db_manager.get_unprocessed_posts(limit=100)
                    
                    if posts:
                        # Filter posts by keywords first
                        relevant_posts = [p for p in posts if self.should_analyze_post(p.text)]
                        logger.info(f"Processing batch of {len(posts)} posts ({len(relevant_posts)} passed keyword filter, {len(posts) - len(relevant_posts)} filtered out)")

                        for post in relevant_posts:
                            # Analyze post
                            result = self.analyzer.analyze_post(post)

                            if result:
                                label, confidence, cluster_id = result

                                # Apply label
                                self.apply_label(post, label, confidence, cluster_id)

                                # Save to database
                                self.db_manager.save_labeled_post(
                                    post, label, confidence, cluster_id
                                )

                        # Mark posts as processed
                        self.db_manager.mark_processed([p.uri for p in posts])
                    
                    # Wait before next batch
                    time.sleep(CONFIG['BATCH_PROCESSING_INTERVAL'])
                    
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    time.sleep(5)
        
        self.processing_thread = threading.Thread(target=process_batches, daemon=True)
        self.processing_thread.start()
        logger.info("Started batch processor")
    
    def apply_label(self, post: BlueskyPost, label: str, confidence: float, cluster_id: int):
        """Apply label to a post"""
        try:
            # Get cluster for context
            clusters = self.db_manager.get_clusters()
            cluster = next((c for c in clusters if c.cluster_id == cluster_id), None)

            if not cluster:
                return

            # Create label
            label_data = {
                "$type": "com.atproto.label.defs#label",
                "src": self.service_did,
                "uri": post.uri,
                "val": label,
                "cts": datetime.now().isoformat() + "Z",
                "exp": (datetime.now() + timedelta(days=30)).isoformat() + "Z",  # Expire after 30 days
                "sig": None  # Would need signing in production
            }

            # Additional metadata
            metadata = {
                "confidence": confidence,
                "cluster_theme": cluster.theme,
                "keywords": cluster.keywords
            }

            # In production, this would be signed and sent to label distribution service
            # For now, we'll store it and make it queryable
            logger.info(f"Applied label '{label}' to post {post.uri} (confidence: {confidence:.2f})")

            # Append labeled post to summary file
            self.append_labeled_post(post, label, confidence, cluster)

            # Create a public post about the label (optional, for transparency)
            if confidence >= 0.85:  # Only for high-confidence labels
                self.post_label_notification(post, label, cluster)

        except Exception as e:
            logger.error(f"Failed to apply label: {e}")
    
    def post_label_notification(self, post: BlueskyPost, label: str, cluster: MisinfoCluster):
        """Post a notification about applied label (for transparency)"""
        try:
            notification_text = f"""🔍 Misinformation Alert

A post has been labeled as containing misinformation.

Theme: {cluster.theme}
Confidence: High

This label helps users identify potentially misleading content. Users can subscribe to our labeling service for automatic warnings.

#MisinformationDetection #FactCheck"""
            
            # Truncate if too long
            if len(notification_text) > 300:
                notification_text = notification_text[:297] + "..."
            
            # Create post
            self.client.com.atproto.repo.create_record(
                repo=self.service_did,
                collection="app.bsky.feed.post",
                record={
                    "$type": "app.bsky.feed.post",
                    "text": notification_text,
                    "createdAt": datetime.now().isoformat() + "Z"
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to post notification: {e}")
    
    def query_labels(self, uri: Optional[str] = None, did: Optional[str] = None) -> List[Dict]:
        """Query labels (implements label service API)"""
        cursor = self.db_manager.conn.cursor()
        
        if uri:
            cursor.execute(
                'SELECT * FROM labeled_posts WHERE uri = ?',
                (uri,)
            )
        elif did:
            cursor.execute(
                'SELECT * FROM labeled_posts WHERE author_did = ?',
                (did,)
            )
        else:
            cursor.execute('SELECT * FROM labeled_posts ORDER BY labeled_at DESC LIMIT 100')
        
        labels = []
        for row in cursor.fetchall():
            label = {
                "uri": row[0],
                "cid": row[1],
                "val": row[4],
                "confidence": row[5],
                "cts": row[9]
            }
            labels.append(label)
        
        return labels
    
    def start(self):
        """Start the labeling service"""
        self.running = True
        
        # Authenticate
        self.authenticate()
        
        # Register as labeler
        self.register_labeler()
        
        # Start services
        self.subscribe_to_firehose()
        self.start_batch_processor()
        
        logger.info("Bluesky Label Service started")
    
    def stop(self):
        """Stop the labeling service"""
        self.running = False
        logger.info("Stopping Bluesky Label Service...")
        
        # Wait for threads to finish
        if self.firehose_thread:
            self.firehose_thread.join(timeout=5)
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        logger.info("Bluesky Label Service stopped")


class LabelServiceAPI:
    """REST API for the label service"""
    
    def __init__(self, service: BlueskyLabelService, port: int = 8080):
        self.service = service
        self.port = port
        
        # Simple HTTP server for label queries
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import urllib.parse
        
        class LabelHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                query_params = urllib.parse.parse_qs(parsed.query)
                
                if parsed.path == '/xrpc/com.atproto.label.queryLabels':
                    # Query labels endpoint
                    uri = query_params.get('uri', [None])[0]
                    did = query_params.get('did', [None])[0]
                    
                    labels = self.server.service.query_labels(uri, did)
                    
                    response = {
                        "labels": labels
                    }
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())
                    
                elif parsed.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b'OK')
                    
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                # Suppress default logging
                pass
        
        self.server = HTTPServer(('', port), LabelHandler)
        self.server.service = service
    
    def start(self):
        """Start the API server"""
        logger.info(f"Starting Label Service API on port {self.port}")
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def main():
    """Main entry point"""
    logger.info("Starting Bluesky Misinformation Labeling Service")
    
    # Check for required environment variables
    if not CONFIG['OPENAI_API_KEY']:
        logger.error("OPENAI_API_KEY not found in .env file")
        sys.exit(1)
    
    if not CONFIG['BLUESKY_HANDLE'] or not CONFIG['BLUESKY_PASSWORD']:
        logger.error("BLUESKY_HANDLE and BLUESKY_PASSWORD required in .env file")
        sys.exit(1)
    
    # Initialize services
    db_manager = DatabaseManager(CONFIG['DB_PATH'])
    embedding_service = EmbeddingService(CONFIG['OPENAI_API_KEY'])
    analyzer = MisinformationAnalyzer(embedding_service, db_manager)
    
    # Load and process Community Notes
    logger.info("Loading Community Notes data...")
    notes = analyzer.load_community_notes(
        'notes-00000.tsv',
        max_notes=CONFIG['MAX_NOTES_FOR_TESTING']
    )
    
    if not notes:
        logger.error("No Community Notes loaded")
        sys.exit(1)
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    analyzer.generate_embeddings(notes)
    
    # Cluster notes
    logger.info("Clustering notes to identify misinformation themes...")
    clusters = analyzer.cluster_notes(notes)
    
    if not clusters:
        logger.warning("No clusters created - continuing with existing clusters")
        clusters = db_manager.get_clusters()
        analyzer.clusters = clusters
    
    logger.info(f"Identified {len(clusters)} misinformation themes")
    for cluster in clusters[:5]:  # Show first 5 themes
        logger.info(f"  Theme: {cluster.theme}")
        logger.info(f"  Keywords: {', '.join(cluster.keywords[:5])}")
        logger.info(f"  Notes in cluster: {len(cluster.notes)}")
        logger.info("")

    # Initialize Bluesky service
    bluesky_service = BlueskyLabelService(analyzer, db_manager)

    # Build keyword filter for efficient pre-filtering
    bluesky_service.build_keyword_filter(clusters)

    # Write themes summary to file
    bluesky_service.write_themes_summary(clusters)

    # Start API server
    api = LabelServiceAPI(bluesky_service, port=8080)
    api.start()

    # Start the labeling service
    bluesky_service.start()
    
    try:
        # Keep the service running
        while True:
            time.sleep(60)
            
            # Log statistics
            cursor = db_manager.conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM labeled_posts')
            labeled_count = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM processing_queue WHERE processed = 0')
            queue_count = cursor.fetchone()[0]

            filter_rate = (bluesky_service.filtered_count / bluesky_service.processed_count * 100) if bluesky_service.processed_count > 0 else 0
            logger.info(f"Stats - Labeled: {labeled_count}, Queue: {queue_count}, Filtered: {bluesky_service.filtered_count}/{bluesky_service.processed_count} ({filter_rate:.1f}%)")
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested...")
        bluesky_service.stop()
        db_manager.close()
        logger.info("Service stopped")


if __name__ == "__main__":
    main()
