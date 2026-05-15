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
import sys
from pathlib import Path

from car_lense_engine.db import open_db, queue

from .browser import PlaywrightFetcher
from .fetcher import Fetcher
from .politeness import PolicyConfig
from .registry import ParserRegistry
from .runner import run_crawler

DEFAULT_DB = Path("db/crawl.sqlite")


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
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    return parser


def _make_fetcher(*, headless: bool) -> Fetcher:
    """Default fetcher factory; overridden in tests via :func:`main` argument."""
    return PlaywrightFetcher(headless=headless)


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
        Optional callable ``(headless: bool) -> Fetcher``. Tests inject a fake
        fetcher here so the CLI can be exercised without Playwright. Typed as
        ``object`` to keep the public surface unencumbered; the callable is
        invoked with a single ``headless=`` keyword.
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
        log.info(
            "crawl starting: db=%s source=%s policy=[min=%.1f max=%.1f off_peak=%s "
            "idle_exit=%ds] queue=[pending=%d in_progress=%d done=%d failed=%d dead=%d]",
            db_path,
            args.source or "<all>",
            policy.min_delay_seconds,
            policy.max_delay_seconds,
            policy.off_peak_only,
            policy.idle_exit_seconds,
            q_stats.pending,
            q_stats.in_progress,
            q_stats.done,
            q_stats.failed,
            q_stats.dead,
        )

        factory = fetcher_factory if fetcher_factory is not None else _make_fetcher
        if not callable(factory):  # pragma: no cover - defensive
            raise TypeError("fetcher_factory must be callable")
        fetcher: Fetcher = factory(headless=args.headless)
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
