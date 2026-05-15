"""Console script for the crawler runtime.

Invoked via the ``crawl`` entry point declared in ``pyproject.toml``::

    crawl [--source SITE_ID] [--db PATH] [--max-items N] [--workers 1]
          [--off-peak] [--headless/--headed] [--min-delay 3.0] [--max-delay 5.0]
          [--idle-exit-seconds 60] [-v]

The CLI does **not** register any parsers — Phase 2 parser packages will
provide a ``register_all(registry)`` helper. When the registry is empty every
claimed item is marked failed-because-no-parser; that's the expected state
during Task 1.5.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from car_lense_engine.db import open_db, queue

from .browser import (
    DEFAULT_NAVIGATION_TIMEOUT_MS,
    DEFAULT_SELECTOR_TIMEOUT_MS,
    DEFAULT_SETTLE_MS,
    DEFAULT_WAIT_UNTIL,
    PlaywrightFetcher,
    WaitUntil,
)
from .curlcffi_fetcher import CurlCffiFetcher
from .fetcher import Fetcher
from .politeness import PolicyConfig
from .proxy import mask_proxy_url, validate_proxy_url
from .registry import ParserRegistry
from .routing import MultiFetcher, known_sources
from .runner import run_crawler

DEFAULT_DB = Path("db/crawl.sqlite")
WAIT_UNTIL_CHOICES: tuple[str, ...] = ("domcontentloaded", "load", "networkidle")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``argparse`` parser for the ``crawl`` command."""
    parser = argparse.ArgumentParser(
        prog="crawl",
        description=(
            "Run the Car Lense crawler against the durable queue. The runtime is "
            "site-agnostic; parsers are registered separately (Phase 2)."
        ),
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="restrict claims to a single source identifier (default: all sources)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"path to the crawler SQLite DB (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="stop after processing N queue items (default: unlimited)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of worker threads (must be 1 for now; future work)",
    )
    parser.add_argument(
        "--off-peak",
        action="store_true",
        help="only crawl during the configured off-peak window",
    )
    headless_group = parser.add_mutually_exclusive_group()
    headless_group.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        help="run the browser headless (default)",
    )
    headless_group.add_argument(
        "--headed",
        dest="headless",
        action="store_false",
        help="run the browser with a visible window (debug)",
    )
    parser.set_defaults(headless=True)
    parser.add_argument(
        "--min-delay",
        type=float,
        default=3.0,
        help="minimum jittered politeness delay in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=5.0,
        help="maximum jittered politeness delay in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--idle-exit-seconds",
        type=int,
        default=60,
        help="exit after the queue has been empty this long (default: 60)",
    )
    parser.add_argument(
        "--wait-until",
        type=str,
        choices=WAIT_UNTIL_CHOICES,
        default=DEFAULT_WAIT_UNTIL,
        help=(
            "Playwright page.goto wait_until signal "
            f"(default: {DEFAULT_WAIT_UNTIL}). Use 'networkidle' for heavy SPAs."
        ),
    )
    parser.add_argument(
        "--settle-ms",
        type=int,
        default=DEFAULT_SETTLE_MS,
        help=(
            "milliseconds to wait after navigation for JS to hydrate "
            f"(default: {DEFAULT_SETTLE_MS})"
        ),
    )
    parser.add_argument(
        "--navigation-timeout-ms",
        type=int,
        default=DEFAULT_NAVIGATION_TIMEOUT_MS,
        help=(
            "Playwright navigation timeout in milliseconds "
            f"(default: {DEFAULT_NAVIGATION_TIMEOUT_MS})"
        ),
    )
    parser.add_argument(
        "--curl-cffi-sources",
        type=str,
        default="",
        help=(
            "comma-separated source identifiers to route through the curl_cffi "
            "fetcher instead of Playwright (e.g. 'cars_com,hemmings'). Useful "
            "for sites that detect headless-Chromium TLS fingerprints. "
            "Default: empty (all sources use Playwright)."
        ),
    )
    parser.add_argument(
        "--wait-for-selector",
        action="append",
        default=None,
        metavar="SOURCE=SELECTOR",
        help=(
            "register a CSS selector to wait for after navigation when the "
            "fetched URL belongs to SOURCE (e.g. "
            "autotrader='[data-cmp=\"inventoryListing\"]'). Repeatable. The "
            "selector is passed to Playwright's page.wait_for_selector; a "
            "comma-separated CSS list is accepted. On timeout the fetch "
            "falls through to the regular settle delay (it does NOT fail)."
        ),
    )
    parser.add_argument(
        "--selector-timeout-ms",
        type=int,
        default=DEFAULT_SELECTOR_TIMEOUT_MS,
        help=(
            "milliseconds to wait for any --wait-for-selector hint before "
            f"giving up (default: {DEFAULT_SELECTOR_TIMEOUT_MS})."
        ),
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        metavar="URL",
        help=(
            "route ALL crawler traffic through a residential proxy. Accepts "
            "http://, https://, socks4://, socks5:// with optional user:pass@ "
            "credentials (e.g. 'http://user:pass@gate.smartproxy.com:7000'). "
            "Applies to both PlaywrightFetcher and CurlCffiFetcher. Falls back "
            "to the PROXY_URL environment variable when the flag is omitted."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    return parser


def _make_fetcher(
    *,
    headless: bool,
    wait_until: WaitUntil = DEFAULT_WAIT_UNTIL,
    settle_ms: int = DEFAULT_SETTLE_MS,
    navigation_timeout_ms: int = DEFAULT_NAVIGATION_TIMEOUT_MS,
    curl_cffi_sources: tuple[str, ...] = (),
    wait_for_selector_by_source: dict[str, str] | None = None,
    selector_timeout_ms: int = DEFAULT_SELECTOR_TIMEOUT_MS,
    proxy: str | None = None,
) -> Fetcher:
    """Default fetcher factory; overridden in tests via :func:`main` argument.

    When ``curl_cffi_sources`` is non-empty, wraps the Playwright fetcher in a
    :class:`MultiFetcher` that routes the named sources through a shared
    :class:`CurlCffiFetcher` and leaves everything else on Playwright.

    ``proxy`` (when set) is applied to BOTH inner fetchers so every request —
    Playwright-driven or curl_cffi-driven — egresses through the same proxy.
    """
    playwright = PlaywrightFetcher(
        headless=headless,
        wait_until=wait_until,
        settle_ms=settle_ms,
        navigation_timeout_ms=navigation_timeout_ms,
        wait_for_selector_by_source=wait_for_selector_by_source,
        selector_timeout_ms=selector_timeout_ms,
        proxy=proxy,
    )
    if not curl_cffi_sources:
        return playwright
    curl = CurlCffiFetcher(proxy=proxy)
    per_source: dict[str, Fetcher] = dict.fromkeys(curl_cffi_sources, curl)
    return MultiFetcher(per_source=per_source, default=playwright)


def _parse_curl_cffi_sources(raw: str) -> tuple[str, ...]:
    """Split the ``--curl-cffi-sources`` CSV into a tuple of source IDs."""
    if not raw or not raw.strip():
        return ()
    parts = [token.strip() for token in raw.split(",")]
    return tuple(part for part in parts if part)


def _parse_wait_for_selector_args(
    raw_items: list[str] | None,
) -> dict[str, str]:
    """Parse repeated ``--wait-for-selector source=selector`` flags.

    Returns a dict keyed by source identifier. Raises :class:`ValueError` if
    any entry is missing ``=`` or has an empty source / selector. The CLI
    layer wraps that into ``parser.error`` for a clean exit code 2.
    """
    if not raw_items:
        return {}
    result: dict[str, str] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"--wait-for-selector: expected 'source=selector', got {item!r}")
        source, _, selector = item.partition("=")
        source = source.strip()
        selector = selector.strip()
        if not source:
            raise ValueError(f"--wait-for-selector: empty source in {item!r}")
        if not selector:
            raise ValueError(f"--wait-for-selector: empty selector for source {source!r}")
        result[source] = selector
    return result


def main(
    argv: list[str] | None = None,
    *,
    fetcher_factory: object | None = None,
) -> int:
    """Entry point for the ``crawl`` console script.

    Parameters
    ----------
    argv:
        Command-line arguments (defaults to ``sys.argv[1:]``).
    fetcher_factory:
        Optional callable accepting at least ``headless: bool`` and optionally
        ``wait_until``, ``settle_ms``, ``navigation_timeout_ms`` keywords. Tests
        inject a fake fetcher here so the CLI can be exercised without
        Playwright. Typed as ``object`` to keep the public surface
        unencumbered; the callable is invoked with all configured fetcher
        keyword arguments.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    if args.workers != 1:
        parser.error(f"--workers must be 1 (got {args.workers}); multi-worker support is a TODO.")

    if args.min_delay < 0 or args.max_delay < args.min_delay:
        parser.error(
            "invalid delay window: --min-delay must be >= 0 and --max-delay must be >= --min-delay"
        )

    curl_cffi_sources = _parse_curl_cffi_sources(args.curl_cffi_sources)
    if curl_cffi_sources:
        valid_sources = known_sources()
        unknown = [s for s in curl_cffi_sources if s not in valid_sources]
        if unknown:
            parser.error(
                f"--curl-cffi-sources: unknown source(s) {unknown}; "
                f"valid choices are {sorted(valid_sources)}"
            )

    try:
        wait_for_selector_by_source = _parse_wait_for_selector_args(args.wait_for_selector)
    except ValueError as exc:
        parser.error(str(exc))
    if wait_for_selector_by_source:
        valid_sources = known_sources()
        unknown = [s for s in wait_for_selector_by_source if s not in valid_sources]
        if unknown:
            parser.error(
                f"--wait-for-selector: unknown source(s) {unknown}; "
                f"valid choices are {sorted(valid_sources)}"
            )

    if args.selector_timeout_ms <= 0:
        parser.error(f"--selector-timeout-ms must be > 0, got {args.selector_timeout_ms}")

    # Resolve the proxy URL with the documented precedence: --proxy flag wins,
    # otherwise PROXY_URL env var, otherwise no proxy. Validate up front so a
    # bad URL fails with exit code 2 before we open the DB or import Playwright.
    raw_proxy: str | None = args.proxy if args.proxy else os.environ.get("PROXY_URL")
    if raw_proxy is not None and raw_proxy.strip() == "":
        raw_proxy = None
    resolved_proxy: str | None = None
    if raw_proxy is not None:
        try:
            resolved_proxy = validate_proxy_url(raw_proxy)
        except ValueError as exc:
            parser.error(f"--proxy: {exc}")

    db_path: Path = args.db
    if not db_path.exists():
        parser.error(f"DB path does not exist: {db_path}")

    policy = PolicyConfig(
        min_delay_seconds=args.min_delay,
        max_delay_seconds=args.max_delay,
        off_peak_only=args.off_peak,
        idle_exit_seconds=args.idle_exit_seconds,
    )

    conn = open_db(db_path)
    try:
        registry = ParserRegistry()
        if not registry.sources():
            log.warning(
                "no parsers registered — every claimed item will be marked failed. "
                "This is expected for Task 1.5; Phase 2 parser packages will populate "
                "the registry."
            )

        q_stats = queue.stats(conn, source=args.source)
        # IMPORTANT: only log the masked (credentials-free) form of the proxy.
        # The raw URL must never appear in logs.
        proxy_log_repr = mask_proxy_url(resolved_proxy) if resolved_proxy is not None else "<none>"
        log.info(
            "crawl starting: db=%s source=%s policy=[min=%.1f max=%.1f off_peak=%s "
            "idle_exit=%ds] proxy=%s queue=[pending=%d in_progress=%d done=%d failed=%d dead=%d]",
            db_path,
            args.source or "<all>",
            policy.min_delay_seconds,
            policy.max_delay_seconds,
            policy.off_peak_only,
            policy.idle_exit_seconds,
            proxy_log_repr,
            q_stats.pending,
            q_stats.in_progress,
            q_stats.done,
            q_stats.failed,
            q_stats.dead,
        )

        factory = fetcher_factory if fetcher_factory is not None else _make_fetcher
        if not callable(factory):  # pragma: no cover - defensive
            raise TypeError("fetcher_factory must be callable")
        fetcher: Fetcher = factory(
            headless=args.headless,
            wait_until=args.wait_until,
            settle_ms=args.settle_ms,
            navigation_timeout_ms=args.navigation_timeout_ms,
            curl_cffi_sources=curl_cffi_sources,
            wait_for_selector_by_source=wait_for_selector_by_source,
            selector_timeout_ms=args.selector_timeout_ms,
            proxy=resolved_proxy,
        )
        try:
            summary = run_crawler(
                conn=conn,
                fetcher=fetcher,
                registry=registry,
                policy=policy,
                source=args.source,
                max_items=args.max_items,
            )
        finally:
            try:
                fetcher.close()
            except Exception:  # pragma: no cover - defensive
                log.exception("error closing fetcher")
    finally:
        conn.close()

    log.info(
        "crawl exit: reason=%s elapsed=%.1fs requests_total=%d ok=%d failed=%d "
        "listings_inserted=%d urls_enqueued=%d",
        summary.exit_reason,
        summary.elapsed_seconds,
        summary.stats.requests_total,
        summary.stats.requests_succeeded,
        summary.stats.requests_failed,
        summary.stats.listings_inserted,
        summary.stats.urls_enqueued,
    )
    print(
        f"crawl: exit_reason={summary.exit_reason} elapsed={summary.elapsed_seconds:.1f}s "
        f"requests={summary.stats.requests_total} ok={summary.stats.requests_succeeded} "
        f"failed={summary.stats.requests_failed} "
        f"listings_inserted={summary.stats.listings_inserted} "
        f"urls_enqueued={summary.stats.urls_enqueued}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
