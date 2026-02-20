"""Base API client with rate limiting and retry logic."""

import os
import time
import logging
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

BASE_URL = os.getenv("DOME_API_BASE", "https://api.domeapi.io/v1")
API_KEY = os.getenv("DOME_API_KEY")

# Track last request time globally for 1 QPS rate limit
_last_request_time = 0.0


def api_get(endpoint: str, params: dict = None, max_retries: int = 5) -> dict:
    """Make a rate-limited GET request to the Dome API with retry."""
    global _last_request_time

    url = f"{BASE_URL}{endpoint}"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    for attempt in range(max_retries):
        # Rate limit: 1 QPS
        elapsed = time.time() - _last_request_time
        if elapsed < 1.1:
            time.sleep(1.1 - elapsed)

        try:
            _last_request_time = time.time()
            resp = requests.get(url, params=params, headers=headers, timeout=30)

            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                logger.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code >= 500:
                wait = 3 * (attempt + 1)
                logger.warning(f"Server error {resp.status_code}, retry in {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on {endpoint}, attempt {attempt + 1}/{max_retries}")
            time.sleep(2)
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(3)

    raise RuntimeError(f"Failed after {max_retries} retries: {endpoint}")
