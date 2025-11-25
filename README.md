# Bluesky Misinformation Labeling Service

A production-grade misinformation labeling service for Bluesky that uses insights from X/Twitter Community Notes to identify and label misleading content in real-time.

## Features

- **Semantic Analysis**: Uses OpenAI embeddings to understand content meaning
- **Clustering**: Automatically identifies recurring misinformation themes using DBSCAN
- **Real-time Processing**: Subscribes to Bluesky firehose for immediate analysis
- **Stance Detection**: Determines whether posts support, refute, or discuss claims
- **Confidence Scoring**: Labels posts based on configurable confidence thresholds
- **Persistent Storage**: SQLite database for notes, clusters, and labeled posts
- **REST API**: Queryable label service endpoint
- **Caching**: Intelligent embedding cache to reduce API calls

## Architecture

```
Community Notes TSV → Embedding Generation → Clustering → Theme Extraction
                                                              ↓
Bluesky Firehose → Queue → Batch Processor → Semantic Matching → Stance Detection → Label Application
                                                                                            ↓
                                                                                     Label Service API
```

## Installation

1. Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your credentials:
```bash
cp .env.example .env
# Edit .env with your actual credentials
```

3. Place your Community Notes TSV file as `community_notes.tsv` in the same directory

## Configuration

Key settings in the code's CONFIG dictionary:

- `MAX_NOTES_FOR_TESTING`: Limit notes for testing (set to None for production)
- `CONFIDENCE_THRESHOLD`: Minimum confidence for "misinformation" label (default: 0.75)
- `STANCE_DETECTION_THRESHOLD`: Minimum similarity for stance detection (default: 0.7)
- `CLUSTERING_EPS`: DBSCAN epsilon parameter (default: 0.3)
- `BATCH_PROCESSING_INTERVAL`: Seconds between batch processing (default: 30)

## Usage

### Basic Usage (Testing Mode)

```bash
# Run with first 100 Community Notes for testing
python bluesky_labeler.py
```

### Production Mode

1. Set `MAX_NOTES_FOR_TESTING` to `None` in the code or via environment variable
2. Ensure your Bluesky account has proper permissions
3. Run the service:

```bash
python bluesky_labeler.py
```

### API Endpoints

The service exposes a REST API on port 8080:

- `GET /xrpc/com.atproto.label.queryLabels?uri=<uri>` - Query labels for a specific post
- `GET /xrpc/com.atproto.label.queryLabels?did=<did>` - Query labels by author
- `GET /health` - Health check endpoint

## How It Works

### 1. Data Ingestion
- Loads Community Notes from TSV file
- Filters for misleading content with summaries
- Extracts misleading types (factual errors, manipulated media, etc.)

### 2. Embedding Generation
- Uses OpenAI's text-embedding-3-small model
- Caches embeddings locally to reduce API calls
- Batch processing for efficiency

### 3. Clustering
- DBSCAN clustering to group similar misinformation
- Automatic theme extraction using GPT-4
- Keyword identification for each cluster

### 4. Real-time Analysis
- Subscribes to Bluesky firehose
- Queues posts for batch processing
- Generates embeddings for new posts

### 5. Misinformation Detection
- Cosine similarity with cluster centroids
- Stance detection to verify if post supports misinfo
- Confidence-based labeling:
  - `misinformation`: High confidence (≥0.75)
  - `potential-misinformation`: Lower confidence

### 6. Label Application
- Creates structured labels following Bluesky protocol
- Stores in database with metadata
- Optional transparency posts for high-confidence labels

## Database Schema

### community_notes
- Stores processed Community Notes
- Includes embeddings and cluster assignments

### misinfo_clusters
- Misinformation themes with centroids
- Keywords and confidence scores

### labeled_posts
- Posts identified as containing misinformation
- Includes confidence scores and cluster references

### processing_queue
- Queue for posts awaiting analysis
- Tracks processing status

## Testing

The service includes test mode features:

1. **Limited Data**: Set `MAX_NOTES_FOR_TESTING=100` to test with subset
2. **Logging**: Comprehensive logging to file and console
3. **Statistics**: Periodic stats output (labeled posts, queue size)

## Production Considerations

### Performance
- Embedding caching reduces API calls by 50-90%
- Batch processing optimizes throughput
- SQLite for simple deployment (consider PostgreSQL for scale)

### Reliability
- Automatic reconnection to firehose
- Error recovery in all processing loops
- Persistent queue prevents data loss

### Security
- Environment variables for sensitive data
- No storage of raw credentials
- Rate limiting considerations built-in

### Scaling
- Database indexes on frequently queried fields
- Configurable batch sizes
- Separate threads for firehose and processing

## Label Subscription

Users can subscribe to the labeling service through:

1. Direct subscription to the labeler DID
2. Automatic warnings on labeled content
3. Customizable severity settings

## Monitoring

The service provides:
- Detailed logging with rotation
- Periodic statistics output
- Health check endpoint
- Database for historical analysis

## Limitations

1. **Rate Limits**: Respects OpenAI and Bluesky API limits
2. **Language**: Currently optimized for English content
3. **Latency**: Batch processing introduces 30-second delays
4. **Coverage**: Only analyzes public posts

## Future Enhancements

- Multi-language support
- Image/video analysis
- User reporting integration
- Distributed processing
- Machine learning model fine-tuning
- GraphQL API support

## Contributing

Contributions welcome! Key areas:
- Additional language support
- Performance optimizations
- UI for label management
- Enhanced stance detection
- Cross-platform misinformation tracking

## License

MIT License - See LICENSE file for details

## Disclaimer

This service is for educational and research purposes. Always verify information through multiple sources. The service aims to assist users but is not infallible. False positives and false negatives may occur.

## Support

For issues or questions:
1. Check logs in `bluesky_labeler.log`
2. Verify environment variables
3. Ensure Community Notes TSV is properly formatted
4. Test with smaller data subset first

## Acknowledgments

- Community Notes contributors
- OpenAI for embeddings API
- Bluesky/AT Protocol team
- Scikit-learn for clustering algorithms