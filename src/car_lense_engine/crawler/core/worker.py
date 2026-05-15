"""Single-URL worker loop: claim → fetch → parse → persist → mark_done|failed."""

from __future__ import annotations

import logging
import random
import sqlite3
import time
from collections.abc import Callable
from datetime import datetime
from typing import cast

from pydantic import BaseModel, ConfigDict

from car_lense_engine.crawler.parsers.base import (
    DiscoveredUrl,
    ParsedListing,
    ParseResult,
)
from car_lense_engine.db import Listing, Source, listings, queue
from car_lense_engine.db.models import QueueItem

from .fetcher import Fetcher, FetchError
from .politeness import PolicyConfig, jittered_delay
from .registry import ParserRegistry

logger = logging.getLogger(__name__)


class WorkerStats(BaseModel):
    """Aggregate counters tracked by a :class:`Worker` across a run."""

    model_config = ConfigDict(extra="forbid")

    requests_total: int = 0
    requests_succeeded: int = 0
    requests_failed: int = 0
    listings_inserted: int = 0
    urls_enqueued: int = 0


class Worker:
    """Process one queue item at a time. No threads. Politeness sleep after every fetch.

    All side-effecting dependencies (DB connection, fetcher, registry, clock,
    rng, sleep) are injectable so tests can pin them deterministically.
    """

    def __init__(
        self,
        *,
        conn: sqlite3.Connection,
        fetcher: Fetcher,
        registry: ParserRegistry,
        policy: PolicyConfig,
        rng: random.Random | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.conn = conn
        self.fetcher = fetcher
        self.registry = registry
        self.policy = policy
        self.rng = rng if rng is not None else random.Random()
        self.clock = clock if clock is not None else datetime.now
        self._sleep = sleep_fn if sleep_fn is not None else time.sleep
        self.stats = WorkerStats()

    def run_one(self, *, source: str | None = None) -> bool:
        """Process exactly one queue item.

        Returns ``True`` if work was attempted (success or failure), ``False``
        if the queue is empty / no eligible item.
        """
        item = queue.claim_next(self.conn, source=source)
        if item is None:
            return False

        logger.debug(
            "claimed item: source=%s kind=%s url=%s",
            item.source,
            item.kind,
            item.url,
        )

        # Look up parser. Missing parser is a hard failure for this item.
        if not self.registry.has(item.source):
            err = (
                f"no parser registered for source {item.source!r}; "
                f"known sources: {self.registry.sources()}"
            )
            logger.warning("%s — marking failed: url=%s", err, item.url)
            queue.mark_failed(self.conn, item.url, err)
            self._after_attempt(success=False)
            return True

        parser = self.registry.get(item.source)

        # Fetch.
        try:
            page = self.fetcher.fetch(item.url)
        except FetchError as exc:
            err = f"fetch failed: {exc}"
            logger.warning("%s url=%s", err, item.url)
            queue.mark_failed(self.conn, item.url, err)
            self._after_attempt(success=False)
            return True
        except Exception as exc:  # pragma: no cover - defensive
            err = f"unexpected fetch error: {exc!r}"
            logger.exception("unexpected fetch error url=%s", item.url)
            queue.mark_failed(self.conn, item.url, err)
            self._after_attempt(success=False)
            return True

        # Parse.
        try:
            result = parser.parse(
                html=page.html,
                url=page.url,
                kind=item.kind,
                hints=self._build_hints(item),
            )
        except Exception as exc:
            err = f"parse failed: {exc!r}"
            logger.exception("parse failed url=%s", item.url)
            queue.mark_failed(self.conn, item.url, err)
            self._after_attempt(success=False)
            return True

        # Persist.
        try:
            self._persist(result)
        except Exception as exc:
            err = f"persist failed: {exc!r}"
            logger.exception("persist failed url=%s", item.url)
            queue.mark_failed(self.conn, item.url, err)
            self._after_attempt(success=False)
            return True

        queue.mark_done(self.conn, item.url)
        if result.notes:
            logger.info("parser notes (%s): %s", item.url, result.notes)
        self._after_attempt(success=True)
        return True

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _build_hints(item: QueueItem) -> dict[str, str | int | None]:
        return {
            "target_year": item.target_year,
            "target_make": item.target_make,
            "target_model": item.target_model,
        }

    def _persist(self, result: ParseResult) -> None:
        """Apply the parser's :class:`ParseResult` to the DB."""
        if result.new_listing is not None:
            self._insert_listing(result.new_listing)
        for new_url in result.new_urls:
            self._enqueue(new_url)

    def _insert_listing(self, parsed: ParsedListing) -> None:
        listing = Listing(
            listing_id=parsed.listing_id,
            source=cast(Source, parsed.source),
            url=parsed.url,
            year=parsed.year,
            make=parsed.make,
            model=parsed.model,
            trim=parsed.trim,
            body_style=parsed.body_style,
            mileage=parsed.mileage,
            vin=parsed.vin,
            raw_html_sha256=parsed.raw_html_sha256,
        )
        try:
            listings.insert_listing(self.conn, listing)
        except sqlite3.IntegrityError:
            # Already inserted (e.g., revisit). Treat as a no-op success.
            logger.debug("listing already present, skipping insert: %s", parsed.listing_id)
            return
        self.stats.listings_inserted += 1

        # Image URLs from the listing are enqueued as image-kind queue items.
        for image_url in parsed.image_urls:
            self._enqueue(
                DiscoveredUrl(
                    url=image_url,
                    source=parsed.source,
                    kind="image",
                    parent_listing_id=parsed.listing_id,
                )
            )

    def _enqueue(self, du: DiscoveredUrl) -> None:
        inserted = queue.enqueue(
            self.conn,
            url=du.url,
            source=du.source,
            kind=du.kind,
            target_year=du.target_year,
            target_make=du.target_make,
            target_model=du.target_model,
        )
        if inserted:
            self.stats.urls_enqueued += 1

    def _after_attempt(self, *, success: bool) -> None:
        """Update counters and sleep the politeness delay. Runs once per fetched item."""
        self.stats.requests_total += 1
        if success:
            self.stats.requests_succeeded += 1
        else:
            self.stats.requests_failed += 1
        delay = jittered_delay(self.policy, self.rng)
        logger.debug("politeness sleep: %.2fs", delay)
        self._sleep(delay)
