#!/usr/bin/env python3
"""
Community Notes Theme Analysis Over Time

Analyzes how misinformation themes change over time using Community Notes data.
Can generate overall summary reports or specific theme/keyword trend graphs.

Usage:
    python themes_over_time.py                    # Generate overall summary report
    python themes_over_time.py "Climate Change"   # Generate graph for specific theme
    python themes_over_time.py "COVID vaccine"    # Generate graph for specific keywords
"""

import os
import sys
import csv
import json
import pickle
import hashlib
import argparse
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    'EMBEDDING_MODEL': 'text-embedding-3-small',
    'EMBEDDING_BATCH_SIZE': 100,
    'CLUSTERING_EPS': 0.3,
    'CLUSTERING_MIN_SAMPLES': 2,
    'CACHE_DIR': './cache',
    'TIME_WINDOW_DAYS': 30,  # Analyze in 30-day windows
    'TSV_FILE': 'notes-00000.tsv',
    'OUTPUT_SUMMARY': 'themes_timeline_summary.txt',
    'OUTPUT_GRAPH_DIR': './graphs',
}


@dataclass
class CommunityNote:
    """Represents a Community Note with timestamp"""
    note_id: str
    tweet_id: str
    summary: str
    created_at: datetime
    embedding: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None


@dataclass
class ThemeCluster:
    """Represents a theme cluster in a time period"""
    cluster_id: int
    theme: str
    keywords: List[str]
    note_count: int
    time_period: str
    centroid: np.ndarray
    sample_notes: List[str]


class EmbeddingCache:
    """Manages embedding cache"""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, 'embeddings.pkl')
        self.cache = {}
        self._load()

    def _load(self):
        """Load cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"Loaded {len(self.cache)} cached embeddings")
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")

    def _save(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def get(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.cache.get(text_hash)

    def set(self, text: str, embedding: np.ndarray):
        """Cache embedding"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        self.cache[text_hash] = embedding
        self._save()


class ThemeAnalyzer:
    """Analyzes Community Notes themes over time"""

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.embedding_cache = EmbeddingCache(CONFIG['CACHE_DIR'])

    def load_notes_from_tsv(self, tsv_path: str, max_notes: Optional[int] = None) -> List[CommunityNote]:
        """Load Community Notes from TSV file"""
        notes = []

        print(f"Loading Community Notes from {tsv_path}...")

        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')

            for i, row in enumerate(reader):
                if max_notes and i >= max_notes:
                    break

                # Filter for misleading notes with summaries
                if (row['classification'] == 'MISINFORMED_OR_POTENTIALLY_MISLEADING' and
                    row['summary'] and len(row['summary']) > 10):

                    try:
                        # Convert timestamp to datetime
                        created_at = datetime.fromtimestamp(int(row['createdAtMillis']) / 1000)

                        note = CommunityNote(
                            note_id=row['noteId'],
                            tweet_id=row['tweetId'],
                            summary=row['summary'],
                            created_at=created_at
                        )
                        notes.append(note)
                    except (ValueError, KeyError) as e:
                        # Skip notes with invalid timestamps
                        continue

        print(f"Loaded {len(notes)} Community Notes")
        return notes

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text (with caching)"""
        # Check cache first
        cached = self.embedding_cache.get(text)
        if cached is not None:
            return cached

        try:
            response = self.client.embeddings.create(
                input=text,
                model=CONFIG['EMBEDDING_MODEL']
            )
            embedding = np.array(response.data[0].embedding)
            self.embedding_cache.set(text, embedding)
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

    def get_embeddings_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Get embeddings for multiple texts"""
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Check cache first
        for i, text in enumerate(texts):
            cached = self.embedding_cache.get(text)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Get uncached embeddings in batches
        batch_size = CONFIG['EMBEDDING_BATCH_SIZE']
        for i in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[i:i+batch_size]
            batch_indices = uncached_indices[i:i+batch_size]

            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=CONFIG['EMBEDDING_MODEL']
                )

                for idx, embedding_data in zip(batch_indices, response.data):
                    embedding = np.array(embedding_data.embedding)
                    embeddings[idx] = embedding
                    self.embedding_cache.set(texts[idx], embedding)

                print(f"Generated {len(batch)} new embeddings ({i+len(batch)}/{len(uncached_texts)})")

            except Exception as e:
                print(f"Error getting batch embeddings: {e}")

        return embeddings

    def cluster_notes(self, notes: List[CommunityNote]) -> List[ThemeCluster]:
        """Cluster notes to identify themes"""
        # Filter notes with embeddings
        notes_with_embeddings = [n for n in notes if n.embedding is not None]

        if len(notes_with_embeddings) < 2:
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
        clusters_dict = defaultdict(list)
        for note, label in zip(notes_with_embeddings, clustering.labels_):
            if label != -1:  # Ignore noise
                note.cluster_id = label
                clusters_dict[label].append(note)

        return clusters_dict

    def extract_theme_info(self, cluster_notes: List[CommunityNote]) -> Tuple[str, List[str]]:
        """Extract theme description and keywords from notes using GPT"""
        sample_notes = cluster_notes[:5]
        summaries = '\n'.join([f"- {n.summary}" for n in sample_notes])

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing misinformation patterns. Extract the common theme and keywords."
                    },
                    {
                        "role": "user",
                        "content": f"""Analyze these Community Notes and provide:
1. A brief theme description (max 80 characters)
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
            print(f"Error extracting theme: {e}")
            return "Unknown theme", []

    def partition_by_time(self, notes: List[CommunityNote], window_days: int) -> Dict[str, List[CommunityNote]]:
        """Partition notes into time windows"""
        if not notes:
            return {}

        # Sort notes by date
        notes.sort(key=lambda n: n.created_at)

        # Find date range
        min_date = notes[0].created_at
        max_date = notes[-1].created_at

        # Create time windows
        time_partitions = {}
        current_date = min_date

        while current_date <= max_date:
            window_end = current_date + timedelta(days=window_days)
            window_key = f"{current_date.strftime('%Y-%m-%d')} to {window_end.strftime('%Y-%m-%d')}"

            # Get notes in this window
            window_notes = [
                n for n in notes
                if current_date <= n.created_at < window_end
            ]

            if window_notes:
                time_partitions[window_key] = window_notes

            current_date = window_end

        return time_partitions

    def analyze_timeline(self, notes: List[CommunityNote]) -> Dict[str, List[ThemeCluster]]:
        """Analyze themes over time"""
        print(f"\nAnalyzing {len(notes)} notes over time...")

        # Generate embeddings
        print("Generating embeddings...")
        texts = [n.summary for n in notes]
        embeddings = self.get_embeddings_batch(texts)

        for note, embedding in zip(notes, embeddings):
            note.embedding = embedding

        # Partition by time
        time_partitions = self.partition_by_time(notes, CONFIG['TIME_WINDOW_DAYS'])
        print(f"Created {len(time_partitions)} time windows")

        # Cluster each time period
        timeline_themes = {}

        for i, (period, period_notes) in enumerate(time_partitions.items(), 1):
            print(f"\nAnalyzing period {i}/{len(time_partitions)}: {period} ({len(period_notes)} notes)")

            clusters_dict = self.cluster_notes(period_notes)

            if not clusters_dict:
                print(f"  No clusters found for this period")
                continue

            theme_clusters = []

            for cluster_id, cluster_notes in clusters_dict.items():
                # Calculate centroid
                embeddings = np.array([n.embedding for n in cluster_notes])
                centroid = np.mean(embeddings, axis=0)

                # Extract theme
                theme, keywords = self.extract_theme_info(cluster_notes)

                theme_cluster = ThemeCluster(
                    cluster_id=cluster_id,
                    theme=theme,
                    keywords=keywords,
                    note_count=len(cluster_notes),
                    time_period=period,
                    centroid=centroid,
                    sample_notes=[n.summary for n in cluster_notes[:3]]
                )

                theme_clusters.append(theme_cluster)
                print(f"  Cluster {cluster_id}: {theme} ({len(cluster_notes)} notes)")

            timeline_themes[period] = theme_clusters

        return timeline_themes

    def generate_summary_report(self, timeline_themes: Dict[str, List[ThemeCluster]]) -> str:
        """Generate overall summary report using GPT"""
        print("\nGenerating summary report...")

        # Prepare data for GPT
        timeline_data = []
        for period, themes in timeline_themes.items():
            period_data = {
                'period': period,
                'themes': [
                    {
                        'theme': t.theme,
                        'keywords': t.keywords,
                        'note_count': t.note_count
                    }
                    for t in sorted(themes, key=lambda x: x.note_count, reverse=True)[:5]
                ]
            }
            timeline_data.append(period_data)

        # Use GPT to structure the analysis
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert analyst of misinformation trends. Create a clear, insightful summary of how misinformation themes have changed over time."
                    },
                    {
                        "role": "user",
                        "content": f"""Analyze this timeline of Community Notes themes and create a comprehensive summary report.

Timeline Data:
{json.dumps(timeline_data, indent=2)}

Create a report with:
1. Executive Summary (2-3 paragraphs)
2. Major Themes Over Time (identify recurring themes and new emergent themes)
3. Trends and Patterns (how has misinformation evolved?)
4. Notable Observations (any spikes, changes, or interesting patterns)

Make it readable, insightful, and well-structured."""
                    }
                ],
                temperature=0.5,
                max_tokens=2000
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Error generating AI summary. See raw data below."

    def search_theme_over_time(self, notes: List[CommunityNote], search_term: str, use_keywords: bool = True) -> Tuple[List[datetime], List[int]]:
        """Search for specific theme/keyword and track activity over time

        Args:
            notes: List of Community Notes to search
            search_term: Term or phrase to search for
            use_keywords: If True, use simple keyword matching (fast). If False, use semantic search (slow but more accurate)
        """
        print(f"\nSearching for theme: '{search_term}'")

        if use_keywords:
            # Simple keyword matching (fast, no API calls needed)
            print("Using keyword-based search (case-insensitive)...")
            search_lower = search_term.lower()
            relevant_notes = [
                note for note in notes
                if search_lower in note.summary.lower()
            ]
            print(f"Found {len(relevant_notes)} notes containing '{search_term}'")
        else:
            # Semantic search using embeddings (slower but finds related content)
            print("Using semantic search with embeddings...")

            # Generate embedding for search term
            search_embedding = self.get_embedding(search_term)
            if search_embedding is None:
                print("Error: Could not generate embedding for search term")
                return [], []

            # Load embeddings from cache, generate only for uncached notes
            print("Loading embeddings for notes...")

            # First pass: check cache
            uncached_notes = []
            cached_count = 0
            for note in notes:
                cached_embedding = self.embedding_cache.get(note.summary)
                if cached_embedding is not None:
                    note.embedding = cached_embedding
                    cached_count += 1
                else:
                    uncached_notes.append(note)

            print(f"  Loaded {cached_count} from cache")

            # Second pass: generate embeddings for uncached notes
            if uncached_notes:
                print(f"  Generating {len(uncached_notes)} new embeddings...")
                texts = [n.summary for n in uncached_notes]
                embeddings = self.get_embeddings_batch(texts)

                for note, embedding in zip(uncached_notes, embeddings):
                    note.embedding = embedding

            # Calculate similarity for each note
            print("Calculating similarity scores...")
            similarities = []
            for note in notes:
                if note.embedding is not None:
                    similarity = cosine_similarity(
                        note.embedding.reshape(1, -1),
                        search_embedding.reshape(1, -1)
                    )[0, 0]
                    similarities.append((note, similarity))

            # Filter notes with similarity > threshold
            threshold = 0.35  # Adjust as needed (lowered for better recall)
            relevant_notes = [
                note for note, score in similarities
                if score > threshold
            ]

            print(f"Found {len(relevant_notes)} relevant notes (similarity > {threshold})")

        if not relevant_notes:
            return [], []

        # Group by week
        weekly_counts = defaultdict(int)

        for note in relevant_notes:
            # Get start of week
            week_start = note.created_at - timedelta(days=note.created_at.weekday())
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            weekly_counts[week_start] += 1

        # Sort by date
        dates = sorted(weekly_counts.keys())
        counts = [weekly_counts[d] for d in dates]

        return dates, counts

    def plot_theme_graph(self, dates: List[datetime], counts: List[int], search_term: str):
        """Generate line graph for theme activity"""
        if not dates:
            print("No data to plot")
            return

        # Create output directory
        os.makedirs(CONFIG['OUTPUT_GRAPH_DIR'], exist_ok=True)

        # Create plot
        plt.figure(figsize=(14, 7))
        plt.plot(dates, counts, marker='o', linewidth=2, markersize=6, color='#2E86AB')

        # Formatting
        plt.title(f'Community Notes Activity: "{search_term}"', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Community Notes', fontsize=12, fontweight='bold')

        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.gcf().autofmt_xdate()

        # Grid
        plt.grid(True, alpha=0.3, linestyle='--')

        # Add stats box
        total_notes = sum(counts)
        max_count = max(counts)
        max_date = dates[counts.index(max_count)]

        stats_text = f'Total Notes: {total_notes}\nPeak: {max_count} notes on {max_date.strftime("%b %d, %Y")}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save
        filename = f"{search_term.replace(' ', '_').lower()}_timeline.png"
        filepath = os.path.join(CONFIG['OUTPUT_GRAPH_DIR'], filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\nGraph saved to: {filepath}")

        # Show
        plt.show()


def write_summary_file(timeline_themes: Dict[str, List[ThemeCluster]], summary_text: str, output_file: str):
    """Write summary report to file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("COMMUNITY NOTES THEMES OVER TIME - ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Analysis Period: {min(timeline_themes.keys())} to {max(timeline_themes.keys())}\n\n")

        f.write("=" * 80 + "\n")
        f.write("AI-GENERATED SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(summary_text)
        f.write("\n\n")

        f.write("=" * 80 + "\n")
        f.write("DETAILED TIMELINE DATA\n")
        f.write("=" * 80 + "\n\n")

        for period, themes in timeline_themes.items():
            f.write(f"\n{'─' * 80}\n")
            f.write(f"TIME PERIOD: {period}\n")
            f.write(f"{'─' * 80}\n\n")
            f.write(f"Total themes identified: {len(themes)}\n\n")

            # Sort by note count
            sorted_themes = sorted(themes, key=lambda x: x.note_count, reverse=True)

            for i, theme in enumerate(sorted_themes, 1):
                f.write(f"Theme #{i}:\n")
                f.write(f"  Title: {theme.theme}\n")
                f.write(f"  Keywords: {', '.join(theme.keywords)}\n")
                f.write(f"  Community Notes: {theme.note_count}\n")
                f.write(f"  Sample Notes:\n")
                for j, note in enumerate(theme.sample_notes, 1):
                    f.write(f"    {j}. {note[:100]}...\n")
                f.write("\n")

    print(f"\nSummary report saved to: {output_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Analyze Community Notes themes over time',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python themes_over_time.py                    # Generate overall summary
  python themes_over_time.py "Climate Change"   # Graph for climate change theme
  python themes_over_time.py "COVID vaccine"    # Graph for COVID vaccine theme
        """
    )
    parser.add_argument('search_term', nargs='?', help='Theme or keywords to search for (optional)')
    parser.add_argument('--max-notes', type=int, help='Maximum number of notes to analyze')
    parser.add_argument('--window-days', type=int, default=30, help='Time window in days (default: 30)')
    parser.add_argument('--semantic', action='store_true', help='Use semantic search instead of keyword matching (slower but finds related content)')

    args = parser.parse_args()

    # Update config
    if args.window_days:
        CONFIG['TIME_WINDOW_DAYS'] = args.window_days

    # Check API key
    if not CONFIG['OPENAI_API_KEY']:
        print("Error: OPENAI_API_KEY not found in .env file")
        sys.exit(1)

    # Check TSV file
    if not os.path.exists(CONFIG['TSV_FILE']):
        print(f"Error: {CONFIG['TSV_FILE']} not found")
        sys.exit(1)

    # Initialize analyzer
    analyzer = ThemeAnalyzer(CONFIG['OPENAI_API_KEY'])

    # Load notes
    notes = analyzer.load_notes_from_tsv(CONFIG['TSV_FILE'], max_notes=args.max_notes)

    if not notes:
        print("Error: No notes loaded")
        sys.exit(1)

    if args.search_term:
        # Search for specific theme and generate graph
        print(f"\n{'=' * 80}")
        print(f"THEME SEARCH MODE: {args.search_term}")
        print(f"{'=' * 80}")

        dates, counts = analyzer.search_theme_over_time(notes, args.search_term, use_keywords=not args.semantic)

        if dates and counts:
            analyzer.plot_theme_graph(dates, counts, args.search_term)
        else:
            print(f"\nNo relevant Community Notes found for '{args.search_term}'")
            print("Try a different search term or lower the similarity threshold")

    else:
        # Generate overall timeline analysis
        print(f"\n{'=' * 80}")
        print(f"TIMELINE ANALYSIS MODE")
        print(f"{'=' * 80}")

        timeline_themes = analyzer.analyze_timeline(notes)

        if not timeline_themes:
            print("Error: No themes identified")
            sys.exit(1)

        # Generate AI summary
        summary_text = analyzer.generate_summary_report(timeline_themes)

        # Write to file
        write_summary_file(timeline_themes, summary_text, CONFIG['OUTPUT_SUMMARY'])

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Summary report: {CONFIG['OUTPUT_SUMMARY']}")
        print(f"Time windows analyzed: {len(timeline_themes)}")
        print(f"Total themes identified: {sum(len(themes) for themes in timeline_themes.values())}")


if __name__ == "__main__":
    main()
