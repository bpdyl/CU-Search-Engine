"""
Configuration settings for the Vertical Search Engine
"""
import os
from dotenv import load_dotenv
load_dotenv()

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
PUBLICATIONS_FILE = os.path.join(DATA_DIR, 'publications.json')
INDEX_FILE = os.path.join(DATA_DIR, 'index.pkl')
CLASSIFIER_MODEL_FILE = os.path.join(DATA_DIR, 'classifier_model.pkl')
VECTORIZER_FILE = os.path.join(DATA_DIR, 'vectorizer.pkl')
CLASSIFICATION_DATA_DIR = os.path.join(DATA_DIR, 'classification_data')

# Crawler settings
CRAWLER_BASE_URL = "https://pureportal.coventry.ac.uk"
CRAWLER_DEPARTMENT_URL = "https://pureportal.coventry.ac.uk/en/organisations/ics-research-centre-for-computational-science-and-mathematical-mo"
CRAWLER_PERSONS_URL = f"{CRAWLER_DEPARTMENT_URL}/persons/"
CRAWLER_DELAY = 2  # Seconds between requests (politeness)
CRAWLER_MAX_AUTHORS = 50  # Maximum number of authors to crawl
CRAWLER_TIMEOUT = 30  # Page load timeout in seconds

# Search settings
SEARCH_RESULTS_LIMIT = 1000
FIELD_WEIGHTS = {
    'title': 3.0,
    'authors': 2.5,
    'keywords': 2.0,
    'year': 1.5,
    'abstract': 1.0
}

# Classification categories
CLASSIFICATION_CATEGORIES = ['Business', 'Entertainment', 'Health']

# Flask settings
FLASK_DEBUG = True
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000

# Admin authentication
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD')  

# Scheduler settings
CRAWL_INTERVAL_DAYS = 2  # Crawl every 2 days
