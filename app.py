"""
Vertical Academic Search Engine - Flask Application

A Google Scholar-like search engine for Coventry University's
Research Centre for Computational Science and Mathematical Modelling.
"""
import os
import json
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session

import config
from indexer.inverted_index import InvertedIndex
from indexer.preprocessor import download_nltk_data
from search.query_processor import QueryProcessor
from crawler.spider import PUREPortalCrawler, get_sample_publications
from classifier.trainer import ClassifierTrainer, save_sample_training_data
from classifier.predictor import DocumentClassifier
from scheduler.crawl_scheduler import get_scheduler
from scheduler.crawl_history import CrawlHistory
from scheduler.crawl_summary import get_crawl_summary

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'ir-search-engine-secret-key-2024'

# Initialize components
inverted_index = InvertedIndex()
query_processor = None
classifier = DocumentClassifier()
crawl_scheduler = get_scheduler()
crawl_history = CrawlHistory()
crawl_summary = get_crawl_summary()

# Crawler log storage
crawler_log = []


def admin_required(f):
    """Decorator to require admin authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            flash('Please log in to access the admin area.', 'warning')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function


def init_app():
    """Initialize the application components."""
    global query_processor

    # Download NLTK data
    print("Downloading NLTK data...")
    download_nltk_data()

    # Ensure data directory exists
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.CLASSIFICATION_DATA_DIR, exist_ok=True)

    # Try to load existing index
    if inverted_index.load():
        print(f"Loaded index with {len(inverted_index)} documents")
    else:
        print("No existing index found. Load sample data or run crawler.")

    # Initialize query processor
    query_processor = QueryProcessor(inverted_index)

    # Try to load classifier
    if classifier.load_model():
        print("Classifier loaded successfully")
    else:
        print("Classifier not found. Train the classifier from Admin page.")

    # Initialize crawl history (creates mock data if needed)
    print("Initializing crawl scheduler...")
    _ = crawl_history.load_history()


def log_message(msg):
    """Add message to crawler log."""
    crawler_log.append(msg)
    print(msg)


@app.context_processor
def inject_stats():
    """Inject index statistics into all templates."""
    return {
        'stats': inverted_index.get_statistics(),
        'is_admin': session.get('admin_logged_in', False)
    }


# ============ Public Routes ============

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/search')
def search():
    """Search publications."""
    query = request.args.get('q', '').strip()
    results = []

    if query:
        raw_results = query_processor.search(query)

        # Convert to template-friendly format
        for doc_id, doc, score in raw_results:
            results.append({
                'doc_id': doc_id,
                'title': doc.get('title', 'Untitled'),
                'authors': doc.get('authors', []),
                'year': doc.get('year', 'N/A'),
                'abstract': doc.get('abstract', ''),
                'keywords': doc.get('keywords', []),
                'publication_link': doc.get('publication_link', ''),
                'author_profiles': doc.get('author_profiles', {}),
                'score': score
            })

    return render_template('search.html', query=query, results=results)


@app.route('/classify', methods=['GET', 'POST'])
def classify():
    """Document classification page."""
    result = None
    text = ''

    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        if text:
            result = classifier.classify(text)

    # Only show model info to admin
    model_info = None
    if session.get('admin_logged_in'):
        model_info = classifier.get_model_info()

    return render_template('classify.html',
                         result=result,
                         text=text,
                         model_info=model_info)


# ============ Admin Authentication ============

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login page."""
    if session.get('admin_logged_in'):
        return redirect(url_for('admin'))

    error = None
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')

        if username == config.ADMIN_USERNAME and password == config.ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            flash('Successfully logged in as admin.', 'success')
            return redirect(url_for('admin'))
        else:
            error = 'Invalid username or password'

    return render_template('admin_login.html', error=error)


@app.route('/admin/logout')
def admin_logout():
    """Admin logout."""
    session.pop('admin_logged_in', None)
    flash('Successfully logged out.', 'info')
    return redirect(url_for('index'))


# ============ Admin Routes (Protected) ============

@app.route('/admin')
@admin_required
def admin():
    """Admin page for system management."""
    classifier_info = classifier.get_model_info()
    log_text = '\n'.join(crawler_log[-100:]) if crawler_log else None
    schedule_info = crawl_scheduler.get_schedule_info()

    return render_template('admin.html',
                         classifier_info=classifier_info,
                         crawler_log=log_text,
                         schedule_info=schedule_info)


@app.route('/admin/crawl-history')
@admin_required
def admin_crawl_history():
    """Crawl history page."""
    history = crawl_history.get_recent_crawls(20)
    stats = crawl_history.get_crawl_statistics()
    schedule_info = crawl_scheduler.get_schedule_info()

    return render_template('crawl_history.html',
                         history=history,
                         stats=stats,
                         schedule_info=schedule_info)


@app.route('/admin/load-sample-data', methods=['POST'])
@admin_required
def load_sample_data():
    """Load sample publication data."""
    global query_processor

    try:
        # Get sample publications
        publications = get_sample_publications()

        # Save to JSON file
        os.makedirs(config.DATA_DIR, exist_ok=True)
        with open(config.PUBLICATIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                'publications': publications,
                'total_publications': len(publications)
            }, f, indent=2)

        # Build index
        inverted_index.build_from_publications(publications)
        inverted_index.save()

        # Reinitialize query processor
        query_processor = QueryProcessor(inverted_index)

        flash(f'Successfully loaded {len(publications)} sample publications!', 'success')

    except Exception as e:
        flash(f'Error loading sample data: {str(e)}', 'danger')

    return redirect(url_for('admin'))


@app.route('/admin/run-crawler', methods=['POST'])
@admin_required
def run_crawler():
    """Run the web crawler."""
    global query_processor, crawler_log

    max_authors = int(request.form.get('max_authors', 20))
    crawler_log = []

    # Track crawl timing
    started_at = datetime.now()
    crawl_status = "completed"
    crawl_errors = []
    crawler = None

    try:
        flash('Crawler started. This may take several minutes...', 'info')

        crawler = PUREPortalCrawler(callback=log_message)
        publications = crawler.crawl(max_authors=max_authors)

        completed_at = datetime.now()

        if publications:
            # Save crawled data
            crawler.save_data()

            # Build index
            inverted_index.build_from_publications(publications)
            inverted_index.save()

            # Reinitialize query processor
            query_processor = QueryProcessor(inverted_index)

            # Get detailed metrics from crawler
            crawl_metrics = crawler.get_crawl_metrics()

            # Record crawl in history (basic)
            crawl_stats = {
                "authors_crawled": len(crawler.author_profiles),
                "publications_found": crawl_metrics.get('total_publications_found', len(publications)),
                "unique_publications": crawl_metrics.get('unique_publications', len(publications)),
                "duplicates_detected": crawl_metrics.get('duplicates_detected', 0),
                "pages_visited": len(crawler.visited_urls)
            }
            crawl_history.create_crawl_record(
                started_at=started_at,
                completed_at=completed_at,
                stats=crawl_stats,
                status=crawl_status,
                errors=crawl_errors,
                trigger="manual"
            )

            # Save detailed crawl summary
            crawl_summary.create_summary(
                started_at=started_at,
                completed_at=completed_at,
                status=crawl_status,
                trigger="manual",
                crawl_metrics=crawl_metrics,
                errors=crawl_errors
            )

            flash(f'Successfully crawled {len(publications)} unique publications '
                  f'({crawl_metrics.get("duplicates_detected", 0)} duplicates skipped)!', 'success')
        else:
            completed_at = datetime.now()
            # Get metrics even for empty crawl
            crawl_metrics = crawler.get_crawl_metrics() if crawler else {}

            # Record failed/empty crawl
            crawl_history.create_crawl_record(
                started_at=started_at,
                completed_at=completed_at,
                stats={"authors_crawled": 0, "publications_found": 0},
                status="completed_with_warnings",
                errors=["No publications found"],
                trigger="manual"
            )

            # Save summary for empty crawl
            crawl_summary.create_summary(
                started_at=started_at,
                completed_at=completed_at,
                status="completed_with_warnings",
                trigger="manual",
                crawl_metrics=crawl_metrics,
                errors=["No publications found"]
            )

            flash('Crawler completed but no publications found.', 'warning')

    except Exception as e:
        completed_at = datetime.now()
        # Get metrics if crawler was initialized
        crawl_metrics = crawler.get_crawl_metrics() if crawler else {}

        # Record failed crawl
        crawl_history.create_crawl_record(
            started_at=started_at,
            completed_at=completed_at,
            stats={},
            status="failed",
            errors=[str(e)],
            trigger="manual"
        )

        # Save summary for failed crawl
        crawl_summary.create_summary(
            started_at=started_at,
            completed_at=completed_at,
            status="failed",
            trigger="manual",
            crawl_metrics=crawl_metrics,
            errors=[str(e)]
        )

        flash(f'Crawler error: {str(e)}', 'danger')

    return redirect(url_for('admin'))


@app.route('/admin/rebuild-index', methods=['POST'])
@admin_required
def rebuild_index():
    """Rebuild the search index from saved data."""
    global query_processor

    try:
        # Load publications from JSON
        if os.path.exists(config.PUBLICATIONS_FILE):
            with open(config.PUBLICATIONS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            publications = data.get('publications', [])

            if publications:
                # Rebuild index
                inverted_index.build_from_publications(publications)
                inverted_index.save()

                # Reinitialize query processor
                query_processor = QueryProcessor(inverted_index)

                flash(f'Index rebuilt with {len(publications)} publications!', 'success')
            else:
                flash('No publications found in data file.', 'warning')
        else:
            flash('No publication data file found. Load sample data first.', 'warning')

    except Exception as e:
        flash(f'Error rebuilding index: {str(e)}', 'danger')

    return redirect(url_for('admin'))


@app.route('/admin/train-classifier', methods=['POST'])
@admin_required
def train_classifier():
    """Train the document classifier."""
    try:
        # Save sample training data if it doesn't exist
        training_file = os.path.join(config.CLASSIFICATION_DATA_DIR, 'labeled_articles.json')
        if not os.path.exists(training_file):
            save_sample_training_data()

        # Load training data
        trainer = ClassifierTrainer()
        texts, labels = trainer.load_training_data()

        if len(texts) < 10:
            flash('Insufficient training data (need at least 10 samples).', 'warning')
            return redirect(url_for('admin'))

        # Train the model
        stats = trainer.train(texts, labels)
        trainer.save_model()

        # Reload classifier
        classifier.load_model()

        accuracy = stats.get('accuracy', 0) * 100
        flash(f'Classifier trained successfully! Accuracy: {accuracy:.1f}%', 'success')

    except Exception as e:
        flash(f'Training error: {str(e)}', 'danger')

    return redirect(url_for('admin'))


# ============ API Endpoints ============

@app.route('/api/search')
def api_search():
    """API endpoint for search."""
    query = request.args.get('q', '').strip()
    limit = int(request.args.get('limit', 20))

    if not query:
        return jsonify({'error': 'Query parameter required', 'results': []})

    raw_results = query_processor.search(query, limit=limit)

    results = []
    for doc_id, doc, score in raw_results:
        results.append({
            'doc_id': doc_id,
            'title': doc.get('title', ''),
            'authors': doc.get('authors', []),
            'year': doc.get('year', ''),
            'abstract': doc.get('abstract', ''),
            'publication_link': doc.get('publication_link', ''),
            'author_profiles': doc.get('author_profiles', {}),
            'score': round(score, 4)
        })

    return jsonify({
        'query': query,
        'count': len(results),
        'results': results
    })


@app.route('/api/classify', methods=['POST'])
def api_classify():
    """API endpoint for classification."""
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'Text field required'})

    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Empty text'})

    result = classifier.classify(text)
    return jsonify(result)


@app.route('/api/stats')
@admin_required
def api_stats():
    """API endpoint for system statistics (admin only)."""
    return jsonify({
        'index': inverted_index.get_statistics(),
        'classifier': classifier.get_model_info(),
        'schedule': crawl_scheduler.get_schedule_info()
    })


@app.route('/api/crawl-summary')
@admin_required
def api_crawl_summary():
    """API endpoint for detailed crawl summaries (admin only)."""
    count = int(request.args.get('count', 10))
    summaries = crawl_summary.get_recent_summaries(count)
    aggregate = crawl_summary.get_aggregate_statistics()

    return jsonify({
        'recent_summaries': summaries,
        'aggregate_statistics': aggregate
    })


@app.route('/api/crawl-summary/<summary_id>')
@admin_required
def api_crawl_summary_detail(summary_id):
    """API endpoint for a specific crawl summary (admin only)."""
    summary = crawl_summary.get_summary_by_id(summary_id)
    if summary:
        return jsonify(summary)
    return jsonify({'error': 'Summary not found'}), 404


@app.route('/admin/crawl-summary')
@admin_required
def admin_crawl_summary():
    """Detailed crawl summary page."""
    recent = crawl_summary.get_recent_summaries(10)
    aggregate = crawl_summary.get_aggregate_statistics()
    latest = crawl_summary.get_latest_summary()

    return render_template('crawl_summary.html',
                         recent_summaries=recent,
                         aggregate=aggregate,
                         latest=latest)


# ============ Error Handlers ============

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return render_template('base.html', error='Page not found'), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return render_template('base.html', error='Server error occurred'), 500


# ============ Main ============

if __name__ == '__main__':
    print("=" * 60)
    print("Coventry University Research Search Engine")
    print("=" * 60)

    # Initialize application
    init_app()

    print(f"\nStarting server at http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    print(f"Admin login: /admin/login (username: {config.ADMIN_USERNAME})")
    print("Press Ctrl+C to stop\n")

    # Run Flask app
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )
