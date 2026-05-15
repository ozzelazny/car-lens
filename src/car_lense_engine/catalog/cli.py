"""Console script for the NHTSA vPIC catalog builder.

Invoke via the ``build-catalog`` entry point declared in ``pyproject.toml``::

    build-catalog [--output PATH] [--years START:END] [--rebuild] [--max-makes N] [-v]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

from .build_catalog import build_catalog, write_catalog
from .cache import JSONFileCache
from .nhtsa_client import NHTSAClient

DEFAULT_OUTPUT = Path("catalog/classes.json")
DEFAULT_CACHE = Path("catalog/cache")
DEFAULT_START_YEAR = 1981


def _parse_years(value: str) -> tuple[int, int]:
    """Parse a ``START:END`` year-range string into an inclusive tuple."""
    if ":" not in value:
        raise argparse.ArgumentTypeError(f"expected START:END (e.g. 1990:2010), got {value!r}")
    raw_start, raw_end = value.split(":", 1)
    try:
        start = int(raw_start)
        end = int(raw_end)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"non-integer year in {value!r}") from exc
    if start > end:
        raise argparse.ArgumentTypeError(f"start year > end year in {value!r}")
    return start, end


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``argparse`` parser for the ``build-catalog`` command."""
    current_year = datetime.now(UTC).year
    parser = argparse.ArgumentParser(
        prog="build-catalog",
        description="Build the canonical NHTSA vPIC (year, make, model) catalog.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--years",
        type=_parse_years,
        default=(DEFAULT_START_YEAR, current_year),
        help=(f"year range as START:END inclusive (default: {DEFAULT_START_YEAR}:{current_year})"),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="invalidate the on-disk HTTP cache before fetching",
    )
    parser.add_argument(
        "--max-makes",
        type=int,
        default=None,
        help="only process the first N makes (sorted by name); for testing",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"HTTP cache directory (default: {DEFAULT_CACHE})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="log every request",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``build-catalog`` console script."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cache = JSONFileCache(args.cache_dir)
    if args.rebuild:
        logging.getLogger(__name__).info("clearing cache at %s", args.cache_dir)
        cache.clear()

    with httpx.Client(timeout=30) as http:
        client = NHTSAClient(http, cache=cache)
        catalog = build_catalog(
            client,
            year_range=args.years,
            max_makes=args.max_makes,
            progress=not args.verbose,
        )

    write_catalog(catalog, args.output)
    print(
        f"wrote {args.output}: "
        f"{catalog.meta.total_makes} makes / "
        f"{catalog.meta.total_models} models / "
        f"{catalog.meta.total_class_entries} (make,model,year) entries"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
