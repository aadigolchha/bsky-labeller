#!/usr/bin/env python3
"""
Test script for the Bluesky Misinformation Labeling Service
"""

import os
import sys
import time
import json
import sqlite3
from datetime import datetime
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import the main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bluesky_labeler import (
    CommunityNote, MisinfoCluster, BlueskyPost,
    DatabaseManager, EmbeddingService, MisinformationAnalyzer,
    BlueskyLabelService, CONFIG
)


class TestDatabaseManager(unittest.TestCase):
    """Test database operations"""
    
    def setUp(self):
        self.db = DatabaseManager(':memory:')  # Use in-memory database for tests
    
    def test_save_and_retrieve_community_note(self):
        note = CommunityNote(
            note_id='test123',
            tweet_id='tweet456',
            classification='MISINFORMED_OR_POTENTIALLY_MISLEADING',
            summary='Test misinformation about climate change',
            misleading_type='factual_error',
            created_at=datetime.now()
        )
        
        self.db.save_community_note(note)
        
        # Verify saved
        cursor = self.db.conn.cursor()
        cursor.execute('SELECT * FROM community_notes WHERE note_id = ?', ('test123',))
        result = cursor.fetchone()
        
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 'test123')
        self.assertEqual(result[3], 'Test misinformation about climate change')
    
    def test_queue_operations(self):
        post = BlueskyPost(
            uri='at://did:123/app.bsky.feed.post/abc',
            cid='cid123',
            author_did='did:123',
            text='Test post content',
            created_at=datetime.now()
        )
        
        # Add to queue
        self.db.add_to_queue(post)
        
        # Get unprocessed
        posts = self.db.get_unprocessed_posts()
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0].uri, post.uri)
        
        # Mark processed
        self.db.mark_processed([post.uri])
        
        # Verify processed
        posts = self.db.get_unprocessed_posts()
        self.assertEqual(len(posts), 0)
    
    def tearDown(self):
        self.db.close()


class TestEmbeddingService(unittest.TestCase):
    """Test embedding generation and caching"""
    
    @patch('openai.OpenAI')
    def setUp(self, mock_openai):
        # Mock OpenAI client
        self.mock_client = MagicMock()
        mock_openai.return_value = self.mock_client
        
        # Create service with mocked client
        self.service = EmbeddingService('fake-api-key')
        self.service.cache_dir = '/tmp/test_cache'
        os.makedirs(self.service.cache_dir, exist_ok=True)
    
    def test_embedding_generation(self):
        # Mock embedding response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=np.random.rand(1536).tolist())]
        self.mock_client.embeddings.create.return_value = mock_response
        
        # Generate embedding
        text = "Test text for embedding"
        embedding = self.service.get_embedding(text)
        
        self.assertIsNotNone(embedding)
        self.assertEqual(len(embedding), 1536)
        
        # Verify caching
        embedding2 = self.service.get_embedding(text)
        np.testing.assert_array_equal(embedding, embedding2)
        
        # Should only call API once due to caching
        self.mock_client.embeddings.create.assert_called_once()
    
    def test_batch_embedding(self):
        # Mock batch response
        mock_embeddings = [np.random.rand(1536).tolist() for _ in range(3)]
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=emb) for emb in mock_embeddings]
        self.mock_client.embeddings.create.return_value = mock_response
        
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = self.service.get_embeddings_batch(texts)
        
        self.assertEqual(len(embeddings), 3)
        for emb in embeddings:
            self.assertEqual(len(emb), 1536)


class TestMisinformationAnalyzer(unittest.TestCase):
    """Test misinformation analysis pipeline"""
    
    def setUp(self):
        # Create test components
        self.db = DatabaseManager(':memory:')
        
        # Mock embedding service
        self.embedding_service = MagicMock()
        self.embedding_service.get_embedding.return_value = np.random.rand(1536)
        self.embedding_service.get_embeddings_batch.return_value = [
            np.random.rand(1536) for _ in range(5)
        ]
        
        self.analyzer = MisinformationAnalyzer(self.embedding_service, self.db)
    
    def test_load_community_notes(self):
        # Create test TSV content matching the actual notes-00000.tsv structure
        tsv_content = """noteId\tnoteAuthorParticipantId\tcreatedAtMillis\ttweetId\tclassification\tbelievable\tharmful\tvalidationDifficulty\tmisleadingOther\tmisleadingFactualError\tmisleadingManipulatedMedia\tmisleadingOutdatedInformation\tmisleadingMissingImportantContext\tmisleadingUnverifiedClaimAsFact\tmisleadingSatire\tnotMisleadingOther\tnotMisleadingFactuallyCorrect\tnotMisleadingOutdatedButNotWhenWritten\tnotMisleadingClearlySatire\tnotMisleadingPersonalOpinion\ttrustworthySources\tsummary\tisMediaNote
note1\tauthor1\t1700000000000\ttweet1\tMISINFORMED_OR_POTENTIALLY_MISLEADING\t\t\t\t0\t1\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t1\tClimate change is a hoax perpetrated by scientists\t0
note2\tauthor2\t1700000001000\ttweet2\tMISINFORMED_OR_POTENTIALLY_MISLEADING\t\t\t\t0\t1\t0\t0\t0\t1\t0\t0\t0\t0\t0\t0\t1\tVaccines contain microchips for tracking\t0
note3\tauthor3\t1700000002000\ttweet3\tNOT_MISLEADING\t\t\t\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t1\tThis is accurate information\t0"""

        # Write test TSV
        with open('/tmp/test_notes.tsv', 'w') as f:
            f.write(tsv_content)

        # Load notes
        notes = self.analyzer.load_community_notes('/tmp/test_notes.tsv', max_notes=10)

        self.assertEqual(len(notes), 2)  # Only misleading notes
        self.assertEqual(notes[0].summary, 'Climate change is a hoax perpetrated by scientists')
        self.assertEqual(notes[0].misleading_type, 'factual_error')
    
    @patch('openai.OpenAI')
    def test_clustering(self, mock_openai):
        # Create test notes with embeddings
        notes = []
        for i in range(10):
            note = CommunityNote(
                note_id=f'note{i}',
                tweet_id=f'tweet{i}',
                classification='MISINFORMED_OR_POTENTIALLY_MISLEADING',
                summary=f'Test misinformation {i % 3}',  # 3 themes
                misleading_type='factual_error',
                created_at=datetime.now(),
                embedding=np.random.rand(1536)
            )
            notes.append(note)
        
        # Mock GPT response for theme extraction
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(
            content='{"theme": "Test theme", "keywords": ["test", "misinformation"]}'
        ))]
        mock_client.chat.completions.create.return_value = mock_response
        
        self.analyzer.openai_client = mock_client
        
        # Perform clustering
        clusters = self.analyzer.cluster_notes(notes)
        
        self.assertGreater(len(clusters), 0)
        for cluster in clusters:
            self.assertIsInstance(cluster.theme, str)
            self.assertIsInstance(cluster.keywords, list)
            self.assertGreater(len(cluster.notes), 0)
    
    def test_post_analysis(self):
        # Create test cluster
        cluster = MisinfoCluster(
            cluster_id=0,
            notes=[],
            centroid=np.random.rand(1536),
            theme='Climate misinformation',
            keywords=['climate', 'hoax', 'scientists'],
            confidence=0.9
        )
        self.analyzer.clusters = [cluster]
        
        # Create test post
        post = BlueskyPost(
            uri='at://test/post',
            cid='cid123',
            author_did='did:test',
            text='Climate change is definitely a hoax created by scientists',
            created_at=datetime.now()
        )
        
        # Mock stance detection
        with patch.object(self.analyzer, '_detect_stance', return_value='supports'):
            # Mock embedding similarity
            self.embedding_service.get_embedding.return_value = cluster.centroid + np.random.rand(1536) * 0.1
            
            result = self.analyzer.analyze_post(post)
            
            self.assertIsNotNone(result)
            label, confidence, cluster_id = result
            self.assertIn(label, ['misinformation', 'potential-misinformation'])
            self.assertGreater(confidence, 0)
            self.assertEqual(cluster_id, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline"""
    
    @patch('openai.OpenAI')
    @patch('atproto.Client')
    def test_full_pipeline(self, mock_bluesky, mock_openai):
        """Test the complete pipeline from Community Notes to labeled posts"""
        
        # Setup mocks
        mock_openai_client = MagicMock()
        mock_openai.return_value = mock_openai_client
        
        # Mock embedding response
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock(embedding=np.random.rand(1536).tolist())]
        mock_openai_client.embeddings.create.return_value = mock_embedding_response
        
        # Mock GPT response
        mock_gpt_response = MagicMock()
        mock_gpt_response.choices = [MagicMock(message=MagicMock(
            content='{"theme": "Climate denial", "keywords": ["climate", "hoax", "conspiracy"]}'
        ))]
        mock_openai_client.chat.completions.create.return_value = mock_gpt_response
        
        # Setup services
        db = DatabaseManager(':memory:')
        embedding_service = EmbeddingService('fake-key')
        analyzer = MisinformationAnalyzer(embedding_service, db)
        
        # Create test Community Note
        note = CommunityNote(
            note_id='test_note_1',
            tweet_id='tweet_123',
            classification='MISINFORMED_OR_POTENTIALLY_MISLEADING',
            summary='Claims that climate change is a hoax perpetrated by scientists for grant money',
            misleading_type='factual_error',
            created_at=datetime.now(),
            embedding=np.random.rand(1536)
        )
        
        # Save note and create cluster
        db.save_community_note(note)
        analyzer.clusters = [
            MisinfoCluster(
                cluster_id=0,
                notes=[note],
                centroid=note.embedding,
                theme='Climate denial conspiracy',
                keywords=['climate', 'hoax', 'scientists', 'conspiracy'],
                confidence=0.9
            )
        ]
        
        # Create test Bluesky post
        post = BlueskyPost(
            uri='at://did:example/app.bsky.feed.post/abc123',
            cid='cid_abc123',
            author_did='did:example',
            text='Just learned that climate change is actually a hoax by scientists!',
            created_at=datetime.now()
        )
        
        # Add to queue
        db.add_to_queue(post)
        
        # Process queue
        posts = db.get_unprocessed_posts()
        self.assertEqual(len(posts), 1)
        
        # Analyze post
        with patch.object(analyzer, '_detect_stance', return_value='supports'):
            result = analyzer.analyze_post(posts[0])
            
            if result:
                label, confidence, cluster_id = result
                db.save_labeled_post(posts[0], label, confidence, cluster_id)
        
        # Verify labeling
        cursor = db.conn.cursor()
        cursor.execute('SELECT * FROM labeled_posts')
        labeled = cursor.fetchall()
        
        self.assertGreater(len(labeled), 0)
        
        # Cleanup
        db.close()


def run_performance_test():
    """Run performance benchmarks"""
    import time
    
    print("\n=== Performance Test ===")
    
    # Test embedding generation speed
    embedding_service = MagicMock()
    embedding_service.get_embeddings_batch.return_value = [
        np.random.rand(1536) for _ in range(100)
    ]
    
    start = time.time()
    texts = [f"Test text {i}" for i in range(100)]
    embeddings = embedding_service.get_embeddings_batch(texts)
    elapsed = time.time() - start
    
    print(f"Generated 100 embeddings in {elapsed:.2f} seconds")
    
    # Test database operations
    db = DatabaseManager(':memory:')
    
    start = time.time()
    for i in range(1000):
        post = BlueskyPost(
            uri=f'at://test/post/{i}',
            cid=f'cid{i}',
            author_did='did:test',
            text=f'Test post {i}',
            created_at=datetime.now()
        )
        db.add_to_queue(post)
    elapsed = time.time() - start
    
    print(f"Added 1000 posts to queue in {elapsed:.2f} seconds")
    
    start = time.time()
    posts = db.get_unprocessed_posts(limit=100)
    elapsed = time.time() - start
    
    print(f"Retrieved 100 posts from queue in {elapsed:.2f} seconds")
    
    db.close()
    print("Performance test completed\n")


def main():
    """Run all tests"""
    print("Running Bluesky Misinformation Labeler Test Suite\n")
    print("=" * 60)
    
    # Check for required files
    print("Checking environment...")

    if not os.path.exists('.env'):
        print("⚠️  Warning: .env file not found. Using example configuration.")
        with open('.env', 'w') as f:
            f.write("OPENAI_API_KEY=test_key\n")
            f.write("BLUESKY_HANDLE=test.bsky.social\n")
            f.write("BLUESKY_PASSWORD=test_password\n")

    # Check for notes file
    if os.path.exists('notes-00000.tsv'):
        print("✅ Found notes-00000.tsv")
    else:
        print("⚠️  Warning: notes-00000.tsv not found in current directory")
    
    # Run unit tests
    print("\nRunning unit tests...")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDatabaseManager))
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingService))
    suite.addTests(loader.loadTestsFromTestCase(TestMisinformationAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run performance test
    if result.wasSuccessful():
        run_performance_test()
    
    # Summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All tests passed!")
        print("\nNext steps:")
        print("1. Add your OpenAI API key to .env file")
        print("2. Add your Bluesky credentials to .env file")
        print("3. Ensure notes-00000.tsv is in the same directory")
        print("4. Update bluesky_labeler.py to use 'notes-00000.tsv' instead of 'community_notes.tsv'")
        print("5. Run: python bluesky_labeler.py")
    else:
        print("❌ Some tests failed. Please review the errors above.")
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())