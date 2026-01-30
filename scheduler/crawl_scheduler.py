"""
Crawl Scheduler

Manages scheduled crawling tasks with configurable intervals.
Supports background execution and incremental index updates.
"""

import os
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Callable
import logging

import config
from .crawl_history import CrawlHistory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrawlScheduler:
    """
    Scheduler for periodic web crawling tasks.

    Runs crawls at specified intervals (default: every 2 days)
    and updates the search index incrementally.
    """

    # Default interval: 2 days in seconds
    DEFAULT_INTERVAL = 2 * 24 * 60 * 60  # 172800 seconds

    def __init__(self, interval_seconds: int = None):
        """
        Initialize the scheduler.

        Args:
            interval_seconds: Time between crawls in seconds.
                            Defaults to 2 days (172800 seconds).
        """
        self.interval = interval_seconds or self.DEFAULT_INTERVAL
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.crawl_callback: Optional[Callable] = None
        self.history = CrawlHistory()
        self.schedule_file = os.path.join(config.DATA_DIR, 'crawl_schedule.json')
        self._load_schedule()

    def _load_schedule(self):
        """Load schedule configuration from file."""
        try:
            if os.path.exists(self.schedule_file):
                with open(self.schedule_file, 'r') as f:
                    data = json.load(f)
                    self.interval = data.get('interval_seconds', self.DEFAULT_INTERVAL)
                    self._next_crawl = data.get('next_crawl')
                    self._enabled = data.get('enabled', True)
            else:
                self._next_crawl = None
                self._enabled = True
                self._save_schedule()
        except (json.JSONDecodeError, IOError):
            self._next_crawl = None
            self._enabled = True

    def _save_schedule(self):
        """Save schedule configuration to file."""
        os.makedirs(os.path.dirname(self.schedule_file), exist_ok=True)
        data = {
            'interval_seconds': self.interval,
            'interval_days': self.interval / (24 * 60 * 60),
            'next_crawl': self._next_crawl,
            'enabled': self._enabled,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.schedule_file, 'w') as f:
            json.dump(data, f, indent=2)

    def set_crawl_callback(self, callback: Callable):
        """
        Set the callback function to execute for each crawl.

        Args:
            callback: Function that performs the actual crawl.
                     Should return a dict with crawl statistics.
        """
        self.crawl_callback = callback

    def get_next_scheduled_crawl(self) -> Optional[datetime]:
        """
        Get the next scheduled crawl time.

        Returns:
            datetime of next scheduled crawl, or None if not scheduled
        """
        if not self._enabled:
            return None

        last_crawl = self.history.get_last_crawl()

        if last_crawl:
            last_time = datetime.fromisoformat(last_crawl['completed_at'])
            next_time = last_time + timedelta(seconds=self.interval)
        else:
            # If no previous crawl, schedule for now + interval
            next_time = datetime.now() + timedelta(seconds=self.interval)

        self._next_crawl = next_time.isoformat()
        self._save_schedule()

        return next_time

    def get_schedule_info(self) -> dict:
        """
        Get current schedule information.

        Returns:
            Dictionary with schedule details
        """
        next_crawl = self.get_next_scheduled_crawl()
        last_crawl = self.history.get_last_crawl()

        now = datetime.now()

        if next_crawl:
            time_until = next_crawl - now
            if time_until.total_seconds() < 0:
                time_until_str = "Overdue"
                is_overdue = True
            else:
                days = time_until.days
                hours = time_until.seconds // 3600
                minutes = (time_until.seconds % 3600) // 60

                if days > 0:
                    time_until_str = f"{days}d {hours}h {minutes}m"
                elif hours > 0:
                    time_until_str = f"{hours}h {minutes}m"
                else:
                    time_until_str = f"{minutes}m"
                is_overdue = False
        else:
            time_until_str = "Not scheduled"
            is_overdue = False

        return {
            "enabled": self._enabled,
            "interval_seconds": self.interval,
            "interval_days": self.interval / (24 * 60 * 60),
            "interval_display": f"Every {self.interval // (24 * 60 * 60)} days",
            "next_crawl": next_crawl.isoformat() if next_crawl else None,
            "next_crawl_display": next_crawl.strftime("%Y-%m-%d %H:%M") if next_crawl else "Not scheduled",
            "time_until_next": time_until_str,
            "is_overdue": is_overdue,
            "last_crawl": last_crawl['completed_at'] if last_crawl else None,
            "last_crawl_display": datetime.fromisoformat(last_crawl['completed_at']).strftime("%Y-%m-%d %H:%M") if last_crawl else "Never",
            "is_running": self.is_running
        }

    def enable(self):
        """Enable scheduled crawling."""
        self._enabled = True
        self._save_schedule()
        logger.info("Crawl scheduler enabled")

    def disable(self):
        """Disable scheduled crawling."""
        self._enabled = False
        self._save_schedule()
        logger.info("Crawl scheduler disabled")

    def set_interval(self, days: float):
        """
        Set the crawl interval.

        Args:
            days: Number of days between crawls
        """
        self.interval = int(days * 24 * 60 * 60)
        self._save_schedule()
        logger.info(f"Crawl interval set to {days} days")

    def start_background_scheduler(self):
        """Start the background scheduler thread."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return

        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("Background scheduler started")

    def stop_background_scheduler(self):
        """Stop the background scheduler."""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Background scheduler stopped")

    def _scheduler_loop(self):
        """Main scheduler loop that runs in background thread."""
        while self.is_running:
            if self._enabled:
                next_crawl = self.get_next_scheduled_crawl()
                now = datetime.now()

                if next_crawl and now >= next_crawl:
                    logger.info("Starting scheduled crawl...")
                    self._execute_crawl(trigger="scheduled")

            # Check every 1 minute
            time.sleep(60)

    def _execute_crawl(self, trigger: str = "manual"):
        """
        Execute a crawl operation.

        Args:
            trigger: What triggered the crawl (manual, scheduled)
        """
        if not self.crawl_callback:
            logger.error("No crawl callback set")
            return None

        started_at = datetime.now()
        logger.info(f"Crawl started at {started_at}")
        self.crawl_callback()

    def trigger_manual_crawl(self):
        """Trigger a manual crawl immediately."""
        return self._execute_crawl(trigger="manual")

    def get_crawl_history(self, limit: int = 10):
        """Get recent crawl history."""
        return self.history.get_recent_crawls(limit)

    def get_history_statistics(self):
        """Get aggregate crawl statistics."""
        return self.history.get_crawl_statistics()


# Singleton instance for the application
_scheduler_instance: Optional[CrawlScheduler] = None


def get_scheduler() -> CrawlScheduler:
    """Get the global scheduler instance."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = CrawlScheduler()
    return _scheduler_instance
