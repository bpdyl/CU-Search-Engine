"""
Web Crawler module for crawling Coventry University PURE portal
"""
from .spider import PUREPortalCrawler
from .robots_parser import RobotsParser

__all__ = ['PUREPortalCrawler', 'RobotsParser']
