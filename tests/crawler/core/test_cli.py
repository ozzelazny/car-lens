"""Tests for the `crawl` CLI."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from car_lense_engine.crawler.core import cli as crawl_cli
from car_lense_engine.crawler.core.fetcher import Fetcher
from car_lense_engine.db import open_db

from .conftest import FakeFetcher


def _fake_factory(**_kw: object) -> Fetcher:
    return FakeFetcher()


def test_cli_help_lists_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        crawl_cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in (
        "--source",
        "--db",
        "--max-items",
        "--workers",
        "--off-peak",
        "--headless",
        "--headed",
        "--min-delay",
        "--max-delay",
        "--idle-exit-seconds",
        "--wait-until",
        "--settle-ms",
        "--navigation-timeout-ms",
    ):
        assert flag in out, f"expected {flag!r} in --help output"


def test_cli_workers_must_be_one(db_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Ensure the DB file exists so we exercise the --workers check, not the path check.
    open_db(db_path).close()
    with pytest.raises(SystemExit) as exc:
        crawl_cli.main(
            ["--workers", "2", "--db", str(db_path)],
            fetcher_factory=_fake_factory,
        )
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--workers must be 1" in err


def test_cli_invalid_db_path_clear_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "does_not_exist.sqlite"
    with pytest.raises(SystemExit) as exc:
        crawl_cli.main(
            ["--db", str(missing)],
            fetcher_factory=_fake_factory,
        )
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "DB path does not exist" in err
    assert str(missing) in err


def test_cli_no_parsers_runs_but_logs_warning(
    db_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Bootstrap an empty DB so the CLI accepts the path.
    open_db(db_path).close()
    caplog.set_level("WARNING")
    rc = crawl_cli.main(
        [
            "--db",
            str(db_path),
            "--idle-exit-seconds",
            "0",
            "--min-delay",
            "0",
            "--max-delay",
            "0",
        ],
        fetcher_factory=_fake_factory,
    )
    assert rc == 0
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "no parsers registered" in messages


def test_cli_invalid_delay_window(db_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    open_db(db_path).close()
    with pytest.raises(SystemExit) as exc:
        crawl_cli.main(
            [
                "--db",
                str(db_path),
                "--min-delay",
                "5.0",
                "--max-delay",
                "2.0",
            ],
            fetcher_factory=_fake_factory,
        )
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid delay window" in err


def test_cli_uses_supplied_fetcher_factory(db_path: Path, tmp_path: Path) -> None:
    """Confirm the CLI threads --headless through to the factory and closes the fetcher."""
    conn = open_db(db_path)
    try:
        # No URLs enqueued, so the loop should idle-exit immediately.
        pass
    finally:
        conn.close()

    captured: dict[str, object] = {}

    def _factory(*, headless: bool, **kwargs: object) -> Fetcher:
        captured["headless"] = headless
        captured.update(kwargs)
        captured["fetcher"] = FakeFetcher()
        return captured["fetcher"]  # type: ignore[return-value]

    rc = crawl_cli.main(
        [
            "--db",
            str(db_path),
            "--headed",
            "--idle-exit-seconds",
            "0",
            "--min-delay",
            "0",
            "--max-delay",
            "0",
        ],
        fetcher_factory=_factory,
    )
    assert rc == 0
    assert captured["headless"] is False
    fetcher = captured["fetcher"]
    assert isinstance(fetcher, FakeFetcher)
    assert fetcher.closed is True


def test_cli_passes_wait_until_to_fetcher_factory(db_path: Path) -> None:
    """--wait-until / --settle-ms / --navigation-timeout-ms must reach the factory."""
    open_db(db_path).close()

    captured: dict[str, object] = {}

    def _factory(**kwargs: object) -> Fetcher:
        captured.update(kwargs)
        return FakeFetcher()

    rc = crawl_cli.main(
        [
            "--db",
            str(db_path),
            "--idle-exit-seconds",
            "0",
            "--min-delay",
            "0",
            "--max-delay",
            "0",
            "--wait-until",
            "networkidle",
            "--settle-ms",
            "7500",
            "--navigation-timeout-ms",
            "60000",
        ],
        fetcher_factory=_factory,
    )
    assert rc == 0
    assert captured["wait_until"] == "networkidle"
    assert captured["settle_ms"] == 7500
    assert captured["navigation_timeout_ms"] == 60000


def test_cli_settle_ms_default_is_3000(db_path: Path) -> None:
    """The default settle_ms exposed to the factory must be 3000."""
    open_db(db_path).close()

    captured: dict[str, object] = {}

    def _factory(**kwargs: object) -> Fetcher:
        captured.update(kwargs)
        return FakeFetcher()

    rc = crawl_cli.main(
        [
            "--db",
            str(db_path),
            "--idle-exit-seconds",
            "0",
            "--min-delay",
            "0",
            "--max-delay",
            "0",
        ],
        fetcher_factory=_factory,
    )
    assert rc == 0
    assert captured["settle_ms"] == 3000
    assert captured["wait_until"] == "domcontentloaded"
    assert captured["navigation_timeout_ms"] == 30000


def test_cli_filter_by_source_processes_only_matching_items(
    db_path: Path,
) -> None:
    """--source must restrict claim_next; items from other sources stay pending."""
    conn = open_db(db_path)
    try:
        from car_lense_engine.db import queue

        queue.enqueue(conn, "https://cars.com/a", source="cars_com", kind="listing")
        queue.enqueue(conn, "https://autotrader.com/a", source="autotrader", kind="listing")
    finally:
        conn.close()

    rc = crawl_cli.main(
        [
            "--db",
            str(db_path),
            "--source",
            "cars_com",
            "--idle-exit-seconds",
            "0",
            "--min-delay",
            "0",
            "--max-delay",
            "0",
        ],
        fetcher_factory=_fake_factory,
    )
    assert rc == 0

    conn2: sqlite3.Connection = open_db(db_path)
    try:
        rows = conn2.execute("SELECT url, status FROM crawl_queue ORDER BY url").fetchall()
        states = {r["url"]: r["status"] for r in rows}
    finally:
        conn2.close()
    # cars_com item processed (no parser → failed). autotrader still pending.
    assert states["https://cars.com/a"] == "failed"
    assert states["https://autotrader.com/a"] == "pending"
