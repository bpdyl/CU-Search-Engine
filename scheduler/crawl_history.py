"""
Crawl History Management

Stores and retrieves crawl history records including mock data for demonstration.
"""

import os
import json
from datetime import datetime, timedelta
import random

import config


class CrawlHistory:
    """Manages crawl history records."""

    def __init__(self):
        """Initialize crawl history manager."""
        self.history_file = os.path.join(config.DATA_DIR, 'crawl_history.json')
        self._ensure_history_file()

    def _ensure_history_file(self):
        """Ensure history file exists with mock data if needed."""
        if not os.path.exists(self.history_file):
            self._create_mock_history()

    def _create_mock_history(self):
        """Create mock crawl history for demonstration."""
        mock_history = self._generate_mock_history()
        self.save_history(mock_history)

    def _generate_mock_history(self):
        """
        Generate realistic mock crawl history data.
        Simulates crawls every 2 days for the past 2 weeks.
        """
        history = []
        now = datetime.now()

        # Generate crawl records for past 14 days (every 2 days = 7 crawls)
        crawl_dates = []
        for i in range(7):
            crawl_date = now - timedelta(days=i * 2)
            crawl_dates.append(crawl_date)

        # Reverse to have oldest first
        crawl_dates.reverse()

        # Base statistics that gradually increase
        base_publications = 142
        base_authors = 45

        for idx, crawl_date in enumerate(crawl_dates):
            # Add some variation to stats
            pubs_found = base_publications + idx * random.randint(2, 5)
            new_pubs = random.randint(0, 4) if idx > 0 else base_publications
            updated_pubs = random.randint(1, 8) if idx > 0 else 0
            authors_crawled = base_authors + random.randint(0, 2)

            # Simulate crawl duration (between 3-8 minutes)
            duration_seconds = random.randint(180, 480)

            # Random success/partial success
            if random.random() > 0.1:  # 90% full success
                status = "completed"
                errors = []
            else:
                status = "completed_with_warnings"
                errors = ["Timeout on 1 author profile", "Retry succeeded after initial failure"]

            record = {
                "id": f"crawl_{crawl_date.strftime('%Y%m%d_%H%M%S')}",
                "started_at": crawl_date.isoformat(),
                "completed_at": (crawl_date + timedelta(seconds=duration_seconds)).isoformat(),
                "duration_seconds": duration_seconds,
                "status": status,
                "statistics": {
                    "authors_crawled": authors_crawled,
                    "publications_found": pubs_found,
                    "new_publications": new_pubs,
                    "updated_publications": updated_pubs,
                    "pages_visited": authors_crawled * 3 + random.randint(5, 15)
                },
                "errors": errors,
                "trigger": "scheduled" if idx > 0 else "initial",
                "index_updated": True
            }
            history.append(record)

        return history

    def save_history(self, history):
        """Save history to file."""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

    def load_history(self):
        """Load history from file."""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def add_crawl_record(self, record):
        """Add a new crawl record to history."""
        history = self.load_history()
        history.append(record)
        # Keep only last 50 records
        if len(history) > 50:
            history = history[-50:]
        self.save_history(history)

    def get_recent_crawls(self, limit=10):
        """Get most recent crawl records."""
        history = self.load_history()
        return sorted(history, key=lambda x: x['started_at'], reverse=True)[:limit]

    def get_last_crawl(self):
        """Get the most recent crawl record."""
        recent = self.get_recent_crawls(1)
        return recent[0] if recent else None

    def get_crawl_statistics(self):
        """Get aggregate statistics from crawl history."""
        history = self.load_history()

        if not history:
            return {
                "total_crawls": 0,
                "successful_crawls": 0,
                "total_publications_discovered": 0,
                "average_duration": 0
            }

        successful = [h for h in history if h['status'] in ['completed', 'completed_with_warnings']]

        total_new_pubs = sum(
            h.get('statistics', {}).get('new_publications', 0)
            for h in history
        )

        avg_duration = sum(h.get('duration_seconds', 0) for h in history) / len(history)

        return {
            "total_crawls": len(history),
            "successful_crawls": len(successful),
            "success_rate": len(successful) / len(history) * 100 if history else 0,
            "total_publications_discovered": total_new_pubs,
            "average_duration_seconds": int(avg_duration),
            "first_crawl": min(h['started_at'] for h in history) if history else None,
            "last_crawl": max(h['started_at'] for h in history) if history else None
        }

    def create_crawl_record(self, started_at, completed_at, stats, status="completed", errors=None, trigger="manual"):
        """
        Create a new crawl record.

        Args:
            started_at: Crawl start datetime
            completed_at: Crawl end datetime
            stats: Dictionary with crawl statistics
            status: Crawl status (completed, failed, completed_with_warnings)
            errors: List of error messages
            trigger: What triggered the crawl (manual, scheduled)

        Returns:
            The created record
        """
        duration = (completed_at - started_at).total_seconds()

        record = {
            "id": f"crawl_{started_at.strftime('%Y%m%d_%H%M%S')}",
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": int(duration),
            "status": status,
            "statistics": stats,
            "errors": errors or [],
            "trigger": trigger,
            "index_updated": status in ['completed', 'completed_with_warnings']
        }

        self.add_crawl_record(record)
        return record


def format_duration(seconds):
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"
