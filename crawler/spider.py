"""
Web Crawler for Coventry University PURE Portal

Uses Selenium to bypass Cloudflare protection and crawl publication data
from the Research Centre for Computational Science and Mathematical Modelling.
"""

import re
import time
import json
import os
from datetime import datetime
from urllib.parse import urljoin, urlparse

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service as EdgeService
from bs4 import BeautifulSoup

try:
    from webdriver_manager.chrome import ChromeDriverManager
    from webdriver_manager.microsoft import EdgeChromiumDriverManager
    USE_WEBDRIVER_MANAGER = True
except ImportError:
    USE_WEBDRIVER_MANAGER = False

from crawler.robots_parser import RobotsParser
import config
crawler_log= []
def log_message(msg):
    """Add message to crawler log."""
    crawler_log.append(msg)
    print(msg)

class PUREPortalCrawler:
    """
    Crawler for Coventry University's PURE Portal.
    Extracts publication data from research centre members.
    """

    def __init__(self, callback=None):
        """
        Initialize the crawler.

        Args:
            callback: Optional callback function for logging messages
        """
        self.callback = callback
        self.driver = None
        self.publications = []
        self.visited_urls = set()
        self.author_profiles = {}
        self.base_url = config.CRAWLER_BASE_URL
        self.robots_parser = RobotsParser(self.base_url)

        # Detailed crawl metrics
        self.crawl_metrics = {
            'total_publications_found': 0,  # All pubs including duplicates
            'unique_publications': 0,  # After deduplication
            'duplicates_detected': 0,  # Number of duplicates skipped
            'publications_per_author': {},  # {author_name: count}
            'unique_publications_per_author': {},  # {author_name: count} after dedup
            'duplicates_per_author': {},  # {author_name: count}
            'publication_authors_map': {},  # {pub_title: [list of authors who have it]}
            'co_authored_publications': 0,  # Publications with multiple authors in our dataset
            'pages_crawled_per_author': {},  # {author_name: page_count}
            'authors_per_publication': {},  # {pub_title: author_count}
        }

    def log(self, message):
        """Log a message via callback or print."""
        if self.callback:
            self.callback(message)
        print(message)

    def init_driver(self):
        """Initialize Selenium WebDriver with Chrome or Edge."""
        # Try Chrome first, then Edge as fallback
        if self._try_init_chrome():
            return True


    def _try_init_chrome(self):
        """Try to initialize Chrome WebDriver."""
        chrome_options = ChromeOptions()

        # Anti-detection settings
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('--disable-images')


        # Performance settings
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')

        chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
        chrome_options.add_experimental_option('useAutomationExtension', False)


        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            
            self.driver.set_page_load_timeout(config.CRAWLER_TIMEOUT)

            self.log("Chrome WebDriver initialized successfully")
            return True

        except Exception as e:
            self.log(f"Chrome initialization failed: {e}")
            return False
        
    def close_driver(self):
        """Close the WebDriver."""
        if self.driver:
            try:
                self.driver.quit()
                self.log("WebDriver closed")
            except Exception:
                pass
        
    def extract_author_profiles(self, soup:BeautifulSoup):
        """
        Extract author profile links from the persons page.

        Args:
            soup: BeautifulSoup object of the persons page

        Returns:
            List of (author_name, profile_url) tuples
        """
        profiles = []

        # Look for person links - PURE portal uses specific patterns
        # Method 1: Look for person list items
        person_items = soup.find_all('div', class_=re.compile(r'rendering|person|result-container'))
        
        for item in person_items:
            link = item.find('a', href=re.compile(r'/en/persons/'))
            if link:
                name = link.get_text(strip=True)
                url = urljoin(self.base_url, link.get('href'))
                if name and url not in [p[1] for p in profiles]:
                    profiles.append((name, url))

        # Method 2: Direct link search
        if not profiles:
            for link in soup.find_all('a', href=re.compile(r'/en/persons/[\w-]+')):
                href = link.get('href', '')
                if '/persons/' in href and '/publications' not in href:
                    name = link.get_text(strip=True)
                    url = urljoin(self.base_url, href)
                    if name and len(name) > 2 and url not in [p[1] for p in profiles]:
                        profiles.append((name, url))

        return profiles[:config.CRAWLER_MAX_AUTHORS]

    def extract_publications_from_profile(self, soup, author_name, profile_url):
        """
        Extract publications from an author's profile page.

        Args:
            soup: BeautifulSoup object of the profile page
            author_name: Name of the author
            profile_url: URL of the author's profile

        Returns:
            List of publication dictionaries
        """
        publications = []
        author_total = 0  # Total publications found for this author
        author_unique = 0  # Unique publications added for this author
        author_duplicates = 0  # Duplicates detected for this author

        # Find publication containers - PURE uses various patterns

        # pub_containers = soup.find_all(['li', 'div', 'article'],
        #     class_=re.compile(r'result-container|portal-list-item|publication|research-output'))
        # commented out the code that was looking for various class name , instead we only go for div with class name
        # "result-container"
        pub_containers = soup.find_all(['li', 'div', 'article'],
            class_=re.compile(r'result-container'))

        for container in pub_containers:
            try:
                pub = self._parse_publication(container, author_name, profile_url)
                if pub and pub.get('title'):
                    author_total += 1
                    self.crawl_metrics['total_publications_found'] += 1

                    pub_title = pub['title'].lower().strip()

                    # Track which authors have this publication
                    if pub_title not in self.crawl_metrics['publication_authors_map']:
                        self.crawl_metrics['publication_authors_map'][pub_title] = []
                    if author_name not in self.crawl_metrics['publication_authors_map'][pub_title]:
                        self.crawl_metrics['publication_authors_map'][pub_title].append(author_name)

                    # Check for duplicates
                    if not self._is_duplicate(pub):
                        publications.append(pub)
                        author_unique += 1
                    else:
                        author_duplicates += 1
                        self.crawl_metrics['duplicates_detected'] += 1

            except Exception as e:
                continue

        # Update per-author metrics
        if author_name not in self.crawl_metrics['publications_per_author']:
            self.crawl_metrics['publications_per_author'][author_name] = 0
            self.crawl_metrics['unique_publications_per_author'][author_name] = 0
            self.crawl_metrics['duplicates_per_author'][author_name] = 0

        self.crawl_metrics['publications_per_author'][author_name] += author_total
        self.crawl_metrics['unique_publications_per_author'][author_name] += author_unique
        self.crawl_metrics['duplicates_per_author'][author_name] += author_duplicates

        return publications

    def _parse_publication(self, container, author_name, profile_url):
        """
        Parse a single publication container.

        Args:
            container: BeautifulSoup element containing publication info
            author_name: Name of the author being crawled
            profile_url: URL of the author's profile

        Returns:
            Publication dictionary or None
        """
        pub = {
            'title': '',
            'authors': [],
            'year': '',
            'abstract': '',
            'keywords': [],
            'publication_link': '',
            'author_profiles': {},
            'crawled_from_author': author_name,
            'crawled_from_profile': profile_url,
            'crawled_at': datetime.now().isoformat()
        }

        # Extract title - look for heading or main link
        title_elem = container.find(['h3', 'h4', 'h2'])
        if not title_elem:
            title_elem = container.find('a', class_=re.compile(r'title|link'))
        if not title_elem:
            title_link = container.find('a', href=re.compile(r'/publications/'))
            if title_link:
                title_elem = title_link

        if title_elem:
            pub['title'] = title_elem.get_text(strip=True)

            # Get publication link
            if title_elem.name == 'a':
                pub['publication_link'] = urljoin(self.base_url, title_elem.get('href', ''))
            else:
                link = title_elem.find('a')
                if link:
                    pub['publication_link'] = urljoin(self.base_url, link.get('href', ''))

        # Extract year
        text = container.get_text()
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            pub['year'] = year_match.group()

        # Extract authors
        
        author_links = container.find_all(
        'a',
        attrs={
            'class': 'link person',
            'href': re.compile(r'/en/persons/')
        }
        )

        for link in author_links:
            name = link.get_text(strip=True)
            url = urljoin(self.base_url, link.get('href', ''))
            if name and name not in pub['authors']:
                pub['authors'].append(name)
                pub['author_profiles'][name] = url
        else:
            # Try to find author links directly
            author_links = container.find_all('a', href=re.compile(r'/en/persons/'))
            for link in author_links:
                name = link.get_text(strip=True)
                url = urljoin(self.base_url, link.get('href', ''))
                if name and len(name) > 2 and name not in pub['authors']:
                    pub['authors'].append(name)
                    pub['author_profiles'][name] = url

        # If no authors found, use the profile author
        if not pub['authors']:
            pub['authors'] = [author_name]
            pub['author_profiles'][author_name] = profile_url

        # Extract abstract from publication link if available
        if pub['publication_link']:
            abstract = self.extract_abstract_from_publication(pub['publication_link'])
            if abstract:
                pub['abstract'] = abstract

        return pub if pub['title'] else None

    def extract_abstract_from_publication(self, publication_url):
        """
        Extract abstract from a publication's detail page.

        Args:
            publication_url: URL of the publication

        Returns:
            Abstract text or empty string if not found
        """
        try:
            soup = self.get_page(publication_url)
            if not soup:
                return ''

            # Look for div with class "content-content publication-content"
            content_div = soup.find('div', class_=re.compile(r'content-content.*publication-content'))
            if not content_div:
                return ''

            # Find h2 with "Abstract" text
            abstract_header = content_div.find('h2', class_='subheader', string='Abstract')
            if not abstract_header:
                return ''

            # Get the next sibling div (or find the rendering div nearby)
            abstract_container = abstract_header.find_next('div', class_=re.compile(r'rendering.*abstractportal'))
            if not abstract_container:
                return ''

            # Extract text from textblock div
            textblock = abstract_container.find('div', class_='textblock')
            if textblock:
                abstract_text = textblock.get_text(strip=True)
                # Remove extra whitespace and limit to 2000 chars
                abstract_text = ' '.join(abstract_text.split())
                return abstract_text[:2000]

            return ''

        except Exception as e:
            self.log(f"    Error extracting abstract from {publication_url}: {e}")
            return ''

    def _is_duplicate(self, pub):
        """Check if publication is already collected."""
        for existing in self.publications:
            if existing['title'].lower() == pub['title'].lower():
                return True
            if existing['publication_link'] and existing['publication_link'] == pub['publication_link']:
                return True
        return False

    def get_next_page_link(self, soup):
        """
        Extract the next page link from pagination.

        Args:
            soup: BeautifulSoup object of the current page

        Returns:
            URL of the next page or None if no next page exists
        """
        # Look for the next page link in pagination
        # Pattern: <li class="next"><a href="..." class="nextLink">Next ›</a></li>
        next_li = soup.find('li', class_='next')
        if next_li:
            next_link = next_li.find('a', class_='nextLink')
            if next_link and next_link.get('href'):
                return urljoin(self.base_url, next_link.get('href'))

        # Alternative pattern: look for any next link
        next_link = soup.find('a', class_='nextLink')
        if next_link and next_link.get('href'):
            return urljoin(self.base_url, next_link.get('href'))

        # Another pattern: look for aria-label containing "Next"
        next_link = soup.find('a', attrs={'aria-label': re.compile(r'Next', re.IGNORECASE)})
        if next_link and next_link.get('href'):
            return urljoin(self.base_url, next_link.get('href'))

        return None

    def crawl_all_author_publications(self, author_name, profile_url, max_pages=10):
        """
        Crawl all publications from an author's profile, handling pagination.

        Args:
            author_name: Name of the author
            profile_url: URL of the author's profile
            max_pages: Maximum number of pages to crawl per author (safety limit)

        Returns:
            List of publication dictionaries
        """
        all_publications = []
        current_url = profile_url
        page_num = 1

        while current_url and page_num <= max_pages:
            self.log(f"    Page {page_num}: {current_url}")

            # Get the page
            soup = self.get_page(current_url)
            if not soup:
                break

            # Extract publications from current page
            pubs = self.extract_publications_from_profile(soup, author_name, profile_url)

            # Track pages per author
            self.crawl_metrics['pages_crawled_per_author'][author_name] = page_num

            if pubs:
                all_publications.extend(pubs)
                self.log(f"    Found {len(pubs)} unique publications on page {page_num}")
            else:
                # No publications found, might be end of list
                if page_num == 1:
                    self.log("    No publications found on profile page")
                break

            # Check for next page
            next_page_url = self.get_next_page_link(soup)

            if next_page_url and next_page_url != current_url:
                current_url = next_page_url
                page_num += 1
            else:
                # No more pages
                break

        if page_num > 1:
            self.log(f"    Total: {len(all_publications)} publications across {page_num} pages")

        return all_publications
    
    def wait_for_page_load(self, timeout=10):
        """Wait for page to fully load."""
        try:
            WebDriverWait(self.driver, timeout).until(
                lambda d: d.execute_script('return document.readyState') == 'complete'
            )
            time.sleep(1)  # Additional wait for dynamic content
        except Exception:
            pass

    def wait_for_cloudflare(self, max_wait=60):
        """
        Wait for Cloudflare verification to complete.

        Args:
            max_wait: Maximum seconds to wait for verification

        Returns:
            True if verification passed, False if timed out
        """
        self.log("  Detecting Cloudflare verification...")
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                page_source = self.driver.page_source.lower()
                page_title = self.driver.title.lower()

                # Check if we're still on Cloudflare page
                cloudflare_indicators = [
                    'checking your browser',
                    'verifying you are human',
                    'just a moment',
                    'cloudflare',
                    'please wait',
                    'ddos protection'
                ]

                is_cloudflare = any(indicator in page_source or indicator in page_title
                                   for indicator in cloudflare_indicators)

                if not is_cloudflare:
                    self.log("  Cloudflare verification passed!")
                    return True

                # Still on Cloudflare page, wait and check again
                self.log(f"  Waiting for Cloudflare... ({int(time.time() - start_time)}s)")
                time.sleep(3)

            except Exception as e:
                self.log(f"  Error checking Cloudflare: {e}")
                time.sleep(2)

        self.log("  WARNING: Cloudflare verification timed out. You may need to solve CAPTCHA manually.")
        return False
    
    def get_page(self, url, is_first_request=False):
        """
        Fetch a page using Selenium.

        Args:
            url: URL to fetch
            is_first_request: If True, wait longer for Cloudflare

        Returns:
            BeautifulSoup object or None if failed
        """
        # Check robots.txt
        if not self.robots_parser.can_fetch(url):
            self.log(f"  Blocked by robots.txt: {url}")
            return None

        try:
            self.driver.get(url)
            self.wait_for_page_load()

            # On first request, wait for Cloudflare verification
            if is_first_request:
                if not self.wait_for_cloudflare(max_wait=90):
                    # Give user a chance to manually solve CAPTCHA
                    self.log("  Please solve the CAPTCHA in the browser window if present...")
                    self.log("  Waiting 30 more seconds for manual verification...")
                    time.sleep(30)

            # Respect crawl delay
            delay = max(config.CRAWLER_DELAY, self.robots_parser.get_crawl_delay())
            time.sleep(delay)

            return BeautifulSoup(self.driver.page_source, 'html.parser')

        except Exception as e:
            self.log(f"  Error fetching {url}: {e}")
            return None
    
    
    def crawl(self, max_authors=None):
        """
        Main crawling method.

        Args:
            max_authors: Maximum number of authors to crawl (default from config)

        Returns:
            List of publication dictionaries
        """
        if max_authors is None:
            max_authors = config.CRAWLER_MAX_AUTHORS

        self.publications = []
        self.visited_urls = set()

        self.log("=" * 60)
        self.log("Starting PURE Portal Crawler")
        self.log("=" * 60)

        # Load robots.txt
        # self.log("\nChecking robots.txt...")
        # self.robots_parser.load()
        # self.log(f"Crawl delay: {self.robots_parser.get_crawl_delay()} seconds")

        # Initialize browser
        if not self.init_driver():
            return []

        try:
            # Step 1: Get author profiles from the department persons page
            self.log(f"\nFetching department persons page...")
            self.log(f"URL: {config.CRAWLER_PERSONS_URL}")

            # soup = self.get_page(config.CRAWLER_PERSONS_URL)
            CRAWLER_DEPARTMENT_URL = 'https://pureportal.coventry.ac.uk/en/organisations/ics-research-centre-for-computational-science-and-mathematical-mo'
            self.driver.get(f"{config.CRAWLER_PERSONS_URL}")
            time.sleep(20)

            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            if not soup:
                self.log("Failed to load persons page")
                return []

            # Extract author profiles
            author_profiles = self.extract_author_profiles(soup)
            self.log(f"Found {len(author_profiles)} author profiles")

            # Step 2: Crawl each author's publications (with pagination support)
            for idx, (author_name, profile_url) in enumerate(author_profiles[:max_authors], 1):
                profile_url = profile_url+'/publications'
                self.log(f"\n[{idx}/{min(len(author_profiles), max_authors)}] Crawling: {author_name}")
                self.log(f"  Profile: {profile_url}")

                if profile_url in self.visited_urls:
                    self.log("  Skipping (already visited)")
                    continue

                self.visited_urls.add(profile_url)

                # Extract all publications with pagination support
                pubs = self.crawl_all_author_publications(author_name, profile_url)

                if pubs:
                    self.publications.extend(pubs)
                    self.log(f"  Total found: {len(pubs)} publications")
                else:
                    self.log("  No publications found")

                # Store author profile info
                self.author_profiles[author_name] = profile_url

            # Finalize metrics
            self._finalize_crawl_metrics()

            # Log detailed summary
            self._log_crawl_summary()

            return self.publications

        finally:
            self.close_driver()

    def _finalize_crawl_metrics(self):
        """Calculate final metrics after crawl completes."""
        self.crawl_metrics['unique_publications'] = len(self.publications)

        # Calculate co-authored publications (pubs that appear for multiple authors in our crawl)
        co_authored = 0
        for pub_title, authors in self.crawl_metrics['publication_authors_map'].items():
            author_count = len(authors)
            self.crawl_metrics['authors_per_publication'][pub_title] = author_count
            if author_count > 1:
                co_authored += 1

        self.crawl_metrics['co_authored_publications'] = co_authored

    def _log_crawl_summary(self):
        """Log a detailed summary of the crawl."""
        self.log("\n" + "=" * 60)
        self.log("CRAWL SUMMARY")
        self.log("=" * 60)

        self.log(f"\n📊 OVERALL METRICS:")
        self.log(f"   Total publications found (raw): {self.crawl_metrics['total_publications_found']}")
        self.log(f"   Unique publications indexed: {self.crawl_metrics['unique_publications']}")
        self.log(f"   Duplicates detected & skipped: {self.crawl_metrics['duplicates_detected']}")
        self.log(f"   Co-authored publications: {self.crawl_metrics['co_authored_publications']}")
        self.log(f"   Authors crawled: {len(self.author_profiles)}")

        self.log(f"\n📝 PER-AUTHOR BREAKDOWN:")
        for author in sorted(self.crawl_metrics['publications_per_author'].keys()):
            total = self.crawl_metrics['publications_per_author'].get(author, 0)
            unique = self.crawl_metrics['unique_publications_per_author'].get(author, 0)
            dups = self.crawl_metrics['duplicates_per_author'].get(author, 0)
            pages = self.crawl_metrics['pages_crawled_per_author'].get(author, 1)
            self.log(f"   {author}:")
            self.log(f"      Total found: {total} | Unique added: {unique} | Duplicates: {dups} | Pages: {pages}")

        # Top co-authored publications
        co_authored_pubs = [(title, len(authors)) for title, authors in
                           self.crawl_metrics['publication_authors_map'].items() if len(authors) > 1]
        if co_authored_pubs:
            co_authored_pubs.sort(key=lambda x: -x[1])
            self.log(f"\n🤝 TOP CO-AUTHORED PUBLICATIONS (in our dataset):")
            for title, count in co_authored_pubs[:5]:
                self.log(f"   - [{count} authors] {title[:60]}...")

        self.log("\n" + "=" * 60)

    def get_crawl_metrics(self):
        """
        Get the crawl metrics dictionary.

        Returns:
            Dictionary containing all crawl metrics
        """
        return self.crawl_metrics
    def save_data(self, filepath=None):
        """
        Save crawled data to JSON file.

        Args:
            filepath: Path to save file (default from config)
        """
        if filepath is None:
            filepath = config.PUBLICATIONS_FILE

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = {
            'crawled_at': datetime.now().isoformat(),
            'total_publications': len(self.publications),
            'total_authors': len(self.author_profiles),
            'source_url': config.CRAWLER_PERSONS_URL,
            'publications': self.publications,
            'author_profiles': self.author_profiles
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.log(f"Data saved to {filepath}")

    def load_data(self, filepath=None):
        """
        Load previously crawled data.

        Args:
            filepath: Path to data file (default from config)

        Returns:
            List of publications or empty list if file not found
        """
        if filepath is None:
            filepath = config.PUBLICATIONS_FILE

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.publications = data.get('publications', [])
            self.author_profiles = data.get('author_profiles', {})

            self.log(f"Loaded {len(self.publications)} publications from {filepath}")
            return self.publications

        except FileNotFoundError:
            self.log(f"Data file not found: {filepath}")
            return []
        except json.JSONDecodeError as e:
            self.log(f"Error parsing data file: {e}")
            return []

def get_sample_publications():
    """
    Get sample publications for demo purposes.

    Returns:
        List of sample publication dictionaries
    """
    return [
        {
            'title': 'Deep Learning Approaches for Natural Language Processing in Healthcare',
            'authors': ['Dr. James Sherwood', 'Prof. Elena Martinez', 'Dr. Amir Hassan'],
            'year': '2024',
            'abstract': 'This paper presents novel deep learning methods for processing medical text and clinical notes, enabling better patient outcome prediction.',
            'keywords': ['deep learning', 'NLP', 'healthcare', 'machine learning'],
            'publication_link': 'https://pureportal.coventry.ac.uk/en/publications/deep-learning-nlp-healthcare-2024',
            'author_profiles': {
                'Dr. James Sherwood': 'https://pureportal.coventry.ac.uk/en/persons/james-sherwood',
                'Prof. Elena Martinez': 'https://pureportal.coventry.ac.uk/en/persons/elena-martinez'
            },
            'crawled_from_author': 'Dr. James Sherwood',
            'crawled_at': datetime.now().isoformat()
        },
        {
            'title': 'Computational Methods for Solving Partial Differential Equations',
            'authors': ['Prof. Michael Chen', 'Dr. Sarah Williams'],
            'year': '2024',
            'abstract': 'Novel computational approaches for efficiently solving complex partial differential equations in physics and engineering applications.',
            'keywords': ['computational mathematics', 'PDEs', 'numerical methods'],
            'publication_link': 'https://pureportal.coventry.ac.uk/en/publications/computational-pde-2024',
            'author_profiles': {
                'Prof. Michael Chen': 'https://pureportal.coventry.ac.uk/en/persons/michael-chen'
            },
            'crawled_from_author': 'Prof. Michael Chen',
            'crawled_at': datetime.now().isoformat()
        },
        {
            'title': 'Machine Learning for Financial Market Prediction',
            'authors': ['Dr. Robert Thompson', 'Dr. Lisa Park', 'Prof. David Kumar'],
            'year': '2023',
            'abstract': 'Application of ensemble machine learning models for stock market trend prediction and risk assessment.',
            'keywords': ['machine learning', 'finance', 'prediction', 'ensemble methods'],
            'publication_link': 'https://pureportal.coventry.ac.uk/en/publications/ml-finance-2023',
            'author_profiles': {
                'Dr. Robert Thompson': 'https://pureportal.coventry.ac.uk/en/persons/robert-thompson'
            },
            'crawled_from_author': 'Dr. Robert Thompson',
            'crawled_at': datetime.now().isoformat()
        },
        {
            'title': 'Neural Network Optimization Techniques for Large-Scale Systems',
            'authors': ['Prof. Elena Martinez', 'Dr. Kevin Zhang'],
            'year': '2023',
            'abstract': 'Advanced optimization algorithms for training deep neural networks on distributed computing systems.',
            'keywords': ['neural networks', 'optimization', 'distributed computing'],
            'publication_link': 'https://pureportal.coventry.ac.uk/en/publications/nn-optimization-2023',
            'author_profiles': {
                'Prof. Elena Martinez': 'https://pureportal.coventry.ac.uk/en/persons/elena-martinez'
            },
            'crawled_from_author': 'Prof. Elena Martinez',
            'crawled_at': datetime.now().isoformat()
        },
        {
            'title': 'Statistical Analysis of Climate Change Data Using Big Data Techniques',
            'authors': ['Dr. Amanda White', 'Prof. James Anderson', 'Dr. Maria Garcia'],
            'year': '2023',
            'abstract': 'Big data analytics applied to climate datasets for identifying long-term environmental patterns and predictions.',
            'keywords': ['big data', 'climate change', 'statistics', 'environmental science'],
            'publication_link': 'https://pureportal.coventry.ac.uk/en/publications/climate-bigdata-2023',
            'author_profiles': {
                'Dr. Amanda White': 'https://pureportal.coventry.ac.uk/en/persons/amanda-white'
            },
            'crawled_from_author': 'Dr. Amanda White',
            'crawled_at': datetime.now().isoformat()
        },
        {
            'title': 'Reinforcement Learning for Autonomous Robot Navigation',
            'authors': ['Dr. Amir Hassan', 'Prof. Jennifer Liu'],
            'year': '2024',
            'abstract': 'Deep reinforcement learning algorithms for enabling autonomous robots to navigate complex environments.',
            'keywords': ['reinforcement learning', 'robotics', 'autonomous systems', 'navigation'],
            'publication_link': 'https://pureportal.coventry.ac.uk/en/publications/rl-robotics-2024',
            'author_profiles': {
                'Dr. Amir Hassan': 'https://pureportal.coventry.ac.uk/en/persons/amir-hassan'
            },
            'crawled_from_author': 'Dr. Amir Hassan',
            'crawled_at': datetime.now().isoformat()
        },
        {
            'title': 'Graph Neural Networks for Social Network Analysis',
            'authors': ['Dr. Kevin Zhang', 'Dr. Sophie Brown', 'Prof. Richard Taylor'],
            'year': '2022',
            'abstract': 'Novel graph neural network architectures for analyzing social network structures and predicting information diffusion.',
            'keywords': ['graph neural networks', 'social networks', 'deep learning'],
            'publication_link': 'https://pureportal.coventry.ac.uk/en/publications/gnn-social-2022',
            'author_profiles': {
                'Dr. Kevin Zhang': 'https://pureportal.coventry.ac.uk/en/persons/kevin-zhang'
            },
            'crawled_from_author': 'Dr. Kevin Zhang',
            'crawled_at': datetime.now().isoformat()
        },
        {
            'title': 'Quantum Computing Algorithms for Cryptographic Applications',
            'authors': ['Prof. Richard Taylor', 'Dr. Emily Watson'],
            'year': '2024',
            'abstract': 'Development of quantum algorithms for next-generation cryptographic systems and security protocols.',
            'keywords': ['quantum computing', 'cryptography', 'algorithms', 'security'],
            'publication_link': 'https://pureportal.coventry.ac.uk/en/publications/quantum-crypto-2024',
            'author_profiles': {
                'Prof. Richard Taylor': 'https://pureportal.coventry.ac.uk/en/persons/richard-taylor'
            },
            'crawled_from_author': 'Prof. Richard Taylor',
            'crawled_at': datetime.now().isoformat()
        },
        {
            'title': 'Computer Vision Techniques for Medical Image Analysis',
            'authors': ['Dr. Sarah Williams', 'Dr. James Sherwood', 'Prof. Michael Chen'],
            'year': '2023',
            'abstract': 'Application of convolutional neural networks for automated diagnosis from medical imaging data.',
            'keywords': ['computer vision', 'medical imaging', 'CNN', 'healthcare'],
            'publication_link': 'https://pureportal.coventry.ac.uk/en/publications/cv-medical-2023',
            'author_profiles': {
                'Dr. Sarah Williams': 'https://pureportal.coventry.ac.uk/en/persons/sarah-williams'
            },
            'crawled_from_author': 'Dr. Sarah Williams',
            'crawled_at': datetime.now().isoformat()
        },
        {
            'title': 'Natural Language Understanding for Conversational AI Systems',
            'authors': ['Prof. Jennifer Liu', 'Dr. Maria Garcia'],
            'year': '2024',
            'abstract': 'Transformer-based models for improving natural language understanding in dialogue systems.',
            'keywords': ['NLU', 'conversational AI', 'transformers', 'dialogue systems'],
            'publication_link': 'https://pureportal.coventry.ac.uk/en/publications/nlu-conversational-2024',
            'author_profiles': {
                'Prof. Jennifer Liu': 'https://pureportal.coventry.ac.uk/en/persons/jennifer-liu'
            },
            'crawled_from_author': 'Prof. Jennifer Liu',
            'crawled_at': datetime.now().isoformat()
        },
        {
            'title': 'Bayesian Methods for Uncertainty Quantification in Scientific Computing',
            'authors': ['Dr. Emily Watson', 'Prof. James Anderson'],
            'year': '2022',
            'abstract': 'Probabilistic approaches for quantifying and propagating uncertainty in computational science models.',
            'keywords': ['Bayesian methods', 'uncertainty quantification', 'scientific computing'],
            'publication_link': 'https://pureportal.coventry.ac.uk/en/publications/bayesian-uq-2022',
            'author_profiles': {
                'Dr. Emily Watson': 'https://pureportal.coventry.ac.uk/en/persons/emily-watson'
            },
            'crawled_from_author': 'Dr. Emily Watson',
            'crawled_at': datetime.now().isoformat()
        },
        {
            'title': 'Data Mining Techniques for Healthcare Analytics',
            'authors': ['Dr. Lisa Park', 'Dr. Sophie Brown'],
            'year': '2023',
            'abstract': 'Application of data mining algorithms for discovering patterns in electronic health records.',
            'keywords': ['data mining', 'healthcare', 'analytics', 'EHR'],
            'publication_link': 'https://pureportal.coventry.ac.uk/en/publications/datamining-healthcare-2023',
            'author_profiles': {
                'Dr. Lisa Park': 'https://pureportal.coventry.ac.uk/en/persons/lisa-park'
            },
            'crawled_from_author': 'Dr. Lisa Park',
            'crawled_at': datetime.now().isoformat()
        }
    ]
