"""HTTP client: rate limiting, on-disk caching, retries."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

DEFAULT_UA = (
    "FencingTracker-Personal-Bot/0.1 "
    "(personal research; contact: vivekf@mit.edu)"
)


class HttpClient:
    def __init__(
        self,
        cache_dir: str | Path = ".cache",
        delay_seconds: float = 1.5,
        user_agent: str = DEFAULT_UA,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay_seconds
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_request_at: float = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.html"

    def _read_cache(self, url: str) -> Optional[str]:
        p = self._cache_path(url)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return None

    def _write_cache(self, url: str, body: str) -> None:
        self._cache_path(url).write_text(body, encoding="utf-8")

    def _wait_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.delay - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def get(self, url: str, *, use_cache: bool = True) -> str:
        if use_cache:
            cached = self._read_cache(url)
            if cached is not None:
                log.debug("CACHE %s", url)
                return cached

        self._wait_rate_limit()
        body = self._fetch_with_retry(url)
        self._last_request_at = time.monotonic()
        self._write_cache(url, body)
        return body

    @retry(
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0),
        reraise=True,
    )
    def _fetch_with_retry(self, url: str) -> str:
        log.info("GET %s", url)
        r = self.session.get(url, timeout=self.timeout)
        if r.status_code >= 500:
            raise requests.ConnectionError(f"HTTP {r.status_code} from {url}")
        r.raise_for_status()
        return r.text
