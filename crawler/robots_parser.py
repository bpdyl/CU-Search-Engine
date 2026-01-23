"""
robots.txt Parser for polite crawling

Respects website crawling policies by parsing and following robots.txt rules.
"""
import requests
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser


class RobotsParser:
    """
    Parser for robots.txt files to ensure polite crawling.
    """

    def __init__(self, base_url, user_agent='*'):
        """
        Initialize the robots.txt parser.

        Args:
            base_url: Base URL of the website
            user_agent: User agent string to check permissions for
        """
        self.base_url = base_url
        self.user_agent = user_agent
        self.parser = RobotFileParser()
        self.crawl_delay = 1  # Default delay in seconds
        self._loaded = False

    # def load(self):
    #     """
    #     Load and parse the robots.txt file.

    #     Returns:
    #         True if successfully loaded, False otherwise
    #     """
    #     try:
    #         parsed = urlparse(self.base_url)
    #         robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    #         print(f'Robots url: {robots_url}')
    #         self.parser.set_url(robots_url)
    #         self.parser.read()

    #         # Get crawl delay if specified
    #         delay = self.parser.crawl_delay(self.user_agent)
    #         if delay:
    #             self.crawl_delay = delay

    #         self._loaded = True
    #         return True

    #     except Exception as e:
    #         print(f"Warning: Could not load robots.txt: {e}")
    #         self._loaded = True  # Continue anyway but with default restrictions
    #         return False

    def load(self):
        """
        Load and parse the robots.txt file using requests to avoid 403 errors.
        """
        try:
            parsed = urlparse(self.base_url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            
            # Use requests with a proper User-Agent header
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(robots_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                # Split the text into lines and parse
                self.parser.parse(response.text.splitlines())
                
                # Get crawl delay
                delay = self.parser.crawl_delay(self.user_agent)
                if delay:
                    self.crawl_delay = delay
                
                self._loaded = True
                return True
            elif response.status_code == 404:
                # If robots.txt doesn't exist, all is allowed
                self._loaded = True
                return True
            else:
                # If we get a 403 or other error, Python's RobotFileParser 
                # defaults to disallowing everything. 
                print(f"Warning: robots.txt returned status {response.status_code}")
                self._loaded = True
                return False

        except Exception as e:
            print(f"Warning: Could not load robots.txt: {e}")
            self._loaded = True
            return False
    def can_fetch(self, url):
        """
        Check if a URL can be fetched according to robots.txt.

        Args:
            url: URL to check

        Returns:
            True if allowed, False otherwise
        """
        if not self._loaded:
            self.load()

        try:
            return self.parser.can_fetch(self.user_agent, url)
        except Exception:
            # If there's an error, assume we can fetch
            return True

    def get_crawl_delay(self):
        """
        Get the recommended crawl delay.

        Returns:
            Crawl delay in seconds
        """
        if not self._loaded:
            self.load()

        return self.crawl_delay

    def get_sitemaps(self):
        """
        Get sitemap URLs from robots.txt.

        Returns:
            List of sitemap URLs
        """
        if not self._loaded:
            self.load()

        try:
            return self.parser.site_maps() or []
        except Exception:
            return []
