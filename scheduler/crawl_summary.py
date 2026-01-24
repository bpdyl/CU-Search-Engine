"""
Crawl Summary Storage

Stores detailed metrics and summaries for each crawl session.
Provides an extended version of crawl history with comprehensive statistics.
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

import config


class CrawlSummary:
    """
    Manages detailed crawl summaries with comprehensive metrics.

    Each crawl creates a summary record containing:
    - Basic crawl info (timing, status, trigger)
    - Overall metrics (total pubs, unique pubs, duplicates)
    - Per-author breakdown (pubs found, unique added, duplicates skipped)
    - Co-authorship analysis
    - Page crawl statistics
    """

    def __init__(self):
        """Initialize the crawl summary manager."""
        self.summary_file = os.path.join(config.DATA_DIR, 'crawl_summary.json')
        self.summaries = []
        self._load_summaries()

    def _load_summaries(self):
        """Load existing summaries from file."""
        if os.path.exists(self.summary_file):
            try:
                with open(self.summary_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.summaries = data.get('summaries', [])
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading crawl summaries: {e}")
                self.summaries = []
        else:
            self.summaries = []

    def _save_summaries(self):
        """Save summaries to file."""
        os.makedirs(os.path.dirname(self.summary_file), exist_ok=True)

        data = {
            'last_updated': datetime.now().isoformat(),
            'total_crawls': len(self.summaries),
            'summaries': self.summaries
        }

        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def create_summary(
        self,
        started_at: datetime,
        completed_at: datetime,
        status: str,
        trigger: str,
        crawl_metrics: Dict[str, Any],
        errors: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a new crawl summary record.

        Args:
            started_at: When the crawl started
            completed_at: When the crawl finished
            status: Crawl status (completed, failed, etc.)
            trigger: What triggered the crawl (manual, scheduled)
            crawl_metrics: Detailed metrics from the crawler
            errors: List of errors encountered

        Returns:
            The created summary record
        """
        duration_seconds = (completed_at - started_at).total_seconds()

        # Build per-author summary
        author_summaries = []
        pubs_per_author = crawl_metrics.get('publications_per_author', {})
        unique_per_author = crawl_metrics.get('unique_publications_per_author', {})
        dups_per_author = crawl_metrics.get('duplicates_per_author', {})
        pages_per_author = crawl_metrics.get('pages_crawled_per_author', {})

        for author in pubs_per_author.keys():
            author_summaries.append({
                'name': author,
                'total_publications_found': pubs_per_author.get(author, 0),
                'unique_publications_added': unique_per_author.get(author, 0),
                'duplicates_skipped': dups_per_author.get(author, 0),
                'pages_crawled': pages_per_author.get(author, 1)
            })

        # Sort by total publications found (descending)
        author_summaries.sort(key=lambda x: -x['total_publications_found'])

        # Build co-authorship analysis
        co_authored_list = []
        pub_authors_map = crawl_metrics.get('publication_authors_map', {})
        for pub_title, authors in pub_authors_map.items():
            if len(authors) > 1:
                co_authored_list.append({
                    'title': pub_title,
                    'shared_by_authors': authors,
                    'author_count': len(authors)
                })

        # Sort by author count (descending)
        co_authored_list.sort(key=lambda x: -x['author_count'])

        summary = {
            'id': f"crawl_{int(started_at.timestamp())}",
            'started_at': started_at.isoformat(),
            'completed_at': completed_at.isoformat(),
            'duration_seconds': duration_seconds,
            'duration_formatted': self._format_duration(duration_seconds),
            'status': status,
            'trigger': trigger,
            'errors': errors or [],

            # Overall metrics
            'overall_metrics': {
                'total_publications_found': crawl_metrics.get('total_publications_found', 0),
                'unique_publications_indexed': crawl_metrics.get('unique_publications', 0),
                'duplicates_detected': crawl_metrics.get('duplicates_detected', 0),
                'co_authored_publications': crawl_metrics.get('co_authored_publications', 0),
                'total_authors_crawled': len(pubs_per_author),
                'total_pages_crawled': sum(pages_per_author.values()) if pages_per_author else 0,
                'deduplication_rate': round(
                    (crawl_metrics.get('duplicates_detected', 0) /
                     max(crawl_metrics.get('total_publications_found', 1), 1)) * 100, 2
                )
            },

            # Per-author breakdown
            'author_summaries': author_summaries,

            # Co-authorship analysis
            'co_authorship_analysis': {
                'total_co_authored': len(co_authored_list),
                'publications': co_authored_list[:20]  # Top 20 to keep file size manageable
            },

            # Statistics
            'statistics': {
                'avg_publications_per_author': round(
                    crawl_metrics.get('total_publications_found', 0) /
                    max(len(pubs_per_author), 1), 2
                ),
                'avg_unique_per_author': round(
                    crawl_metrics.get('unique_publications', 0) /
                    max(len(pubs_per_author), 1), 2
                ),
                'avg_pages_per_author': round(
                    sum(pages_per_author.values()) / max(len(pages_per_author), 1), 2
                ) if pages_per_author else 1.0,
                'most_prolific_author': author_summaries[0] if author_summaries else None,
                'highest_duplicate_author': max(
                    author_summaries,
                    key=lambda x: x['duplicates_skipped'],
                    default=None
                )
            }
        }

        # Add to summaries list
        self.summaries.append(summary)

        # Keep only last 50 summaries to prevent file from growing too large
        if len(self.summaries) > 50:
            self.summaries = self.summaries[-50:]

        # Save to file
        self._save_summaries()

        return summary

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def get_latest_summary(self) -> Optional[Dict[str, Any]]:
        """Get the most recent crawl summary."""
        if self.summaries:
            return self.summaries[-1]
        return None

    def get_recent_summaries(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent N crawl summaries."""
        return self.summaries[-count:][::-1]  # Most recent first

    def get_summary_by_id(self, summary_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific summary by ID."""
        for summary in self.summaries:
            if summary.get('id') == summary_id:
                return summary
        return None

    def get_aggregate_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics across all crawls."""
        if not self.summaries:
            return {
                'total_crawls': 0,
                'total_publications_crawled': 0,
                'total_unique_publications': 0,
                'total_duplicates_detected': 0,
                'average_crawl_duration': 0,
                'success_rate': 0
            }

        total_pubs = sum(
            s.get('overall_metrics', {}).get('total_publications_found', 0)
            for s in self.summaries
        )
        total_unique = sum(
            s.get('overall_metrics', {}).get('unique_publications_indexed', 0)
            for s in self.summaries
        )
        total_dups = sum(
            s.get('overall_metrics', {}).get('duplicates_detected', 0)
            for s in self.summaries
        )
        total_duration = sum(
            s.get('duration_seconds', 0) for s in self.summaries
        )
        successful = sum(
            1 for s in self.summaries if s.get('status') == 'completed'
        )

        return {
            'total_crawls': len(self.summaries),
            'total_publications_crawled': total_pubs,
            'total_unique_publications': total_unique,
            'total_duplicates_detected': total_dups,
            'average_crawl_duration': round(total_duration / len(self.summaries), 2),
            'average_crawl_duration_formatted': self._format_duration(
                total_duration / len(self.summaries)
            ),
            'success_rate': round((successful / len(self.summaries)) * 100, 2),
            'first_crawl': self.summaries[0].get('started_at') if self.summaries else None,
            'last_crawl': self.summaries[-1].get('started_at') if self.summaries else None
        }


# Global instance
_crawl_summary = None


def get_crawl_summary() -> CrawlSummary:
    """Get the global CrawlSummary instance."""
    global _crawl_summary
    if _crawl_summary is None:
        _crawl_summary = CrawlSummary()
    return _crawl_summary
