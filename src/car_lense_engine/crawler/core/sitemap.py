"""Sitemap-walking discovery for sites whose search surfaces are unreachable.

Several listing sites (AutoTrader, Cars & Bids) block their dynamic search /
listing endpoints behind Akamai or Cloudflare interstitials yet still publish
machine-readable sitemap XML at well-known paths. The :class:`SitemapWalker`
implements the standard sitemap protocol (https://www.sitemaps.org/protocol.html):

* ``<urlset>`` documents contain terminal ``<url><loc>...</loc></url>`` entries.
* ``<sitemapindex>`` documents contain ``<sitemap><loc>...</loc></sitemap>``
  entries — each ``<loc>`` points at another sitemap XML and is followed
  recursively, depth-first, in document order.

The walker injects a :class:`Fetcher` (typically :class:`CurlCffiFetcher`) so
the same TLS / proxy posture that clears Cloudflare on listing pages also
applies to sitemap fetches. ``.xml.gz`` URLs are transparently decompressed.

Safety:

* XML is parsed via :mod:`defusedxml.ElementTree` — safe against the standard
  XML entity-expansion attack family even when fetching adversarial sites.
* Two paranoia caps protect the caller from runaway sitemaps: ``max_depth``
  bounds nested-index recursion, ``max_urls`` bounds yield volume. Both have
  generous defaults; callers can tighten them.
"""

from __future__ import annotations

import gzip
import logging
import re
from collections.abc import Iterator

from defusedxml import ElementTree as DefusedET

from .fetcher import Fetcher

logger = logging.getLogger(__name__)


# Tag-name patterns we recognise after namespace stripping. The sitemap
# protocol pins ``urlset`` / ``sitemapindex`` / ``url`` / ``sitemap`` / ``loc``
# but vendor-specific namespaces sometimes prefix them with a vendor prefix
# (``image:loc``, ``video:loc`` etc.). We strip namespace URIs from element
# tags before comparing, but vendor prefixes survive as ``ns:tag`` — the
# regex below catches both cases.
_LOC_TAG_RE = re.compile(r"(?:^|:)loc$", re.IGNORECASE)
_URL_TAG_RE = re.compile(r"(?:^|:)url$", re.IGNORECASE)
_SITEMAP_TAG_RE = re.compile(r"(?:^|:)sitemap$", re.IGNORECASE)
_URLSET_TAG_RE = re.compile(r"(?:^|:)urlset$", re.IGNORECASE)
_SITEMAPINDEX_TAG_RE = re.compile(r"(?:^|:)sitemapindex$", re.IGNORECASE)


def _strip_namespace(tag: str) -> str:
    """Drop the ``{namespace-uri}`` prefix that ElementTree attaches to tags.

    ``{http://www.sitemaps.org/schemas/sitemap/0.9}urlset`` → ``urlset``.
    Tags without a namespace are returned unchanged.
    """
    if tag.startswith("{"):
        end = tag.find("}")
        if end != -1:
            return tag[end + 1 :]
    return tag


def _maybe_decompress(body: str, *, url: str) -> str:
    """Return ``body`` as decoded XML, decompressing gzip when applicable.

    Sitemaps are routinely gzipped (``.xml.gz`` URLs). The :class:`Fetcher`
    Protocol always hands back a ``str`` ``html`` field, but if the upstream
    body was binary gzip the bytes survive in the str via Latin-1 round-trip
    (the way :mod:`curl_cffi` decodes when no charset is announced). We
    detect gzip via the standard ``1f 8b`` magic and decompress when the URL
    or the body shape says so.
    """
    if not body:
        return body
    # Cheap path: only consider decompression when the URL hints at gzip or
    # when the body's first two bytes match the gzip magic. Many text bodies
    # start with ``<?xml`` and we don't want to scan them.
    looks_like_gz_url = url.lower().endswith(".gz")
    raw = body.encode("latin-1", errors="replace")
    magic = raw[:2]
    if not looks_like_gz_url and magic != b"\x1f\x8b":
        return body
    try:
        decompressed = gzip.decompress(raw)
    except OSError as exc:
        logger.debug("sitemap gzip decompress failed for %s: %r", url, exc)
        return body
    # Sitemaps are XML — UTF-8 is the realistic encoding; tolerate stragglers.
    return decompressed.decode("utf-8", errors="replace")


class SitemapWalker:
    """Fetch sitemap XML (or sitemap index) and yield listing URLs.

    Sitemap protocol: https://www.sitemaps.org/protocol.html

    * ``<urlset>`` contains ``<url><loc>...</loc></url>`` entries — terminal.
    * ``<sitemapindex>`` contains ``<sitemap><loc>...</loc></sitemap>``
      entries — each ``<loc>`` points to another sitemap XML and is
      recursed depth-first.

    Uses an injected :class:`Fetcher` (typically :class:`CurlCffiFetcher`) so
    the same proxy / TLS impersonation that beats Cloudflare on listing pages
    applies to sitemap fetches too.
    """

    def __init__(
        self,
        *,
        fetcher: Fetcher,
        max_depth: int = 3,
        max_urls: int = 10_000,
    ) -> None:
        if max_depth < 0:
            raise ValueError(f"max_depth must be >= 0, got {max_depth!r}")
        if max_urls <= 0:
            raise ValueError(f"max_urls must be > 0, got {max_urls!r}")
        self._fetcher = fetcher
        self._max_depth = max_depth
        self._max_urls = max_urls

    # ------------------------------------------------------------ public API

    def walk(self, root_url: str) -> Iterator[str]:
        """Yield every ``<loc>`` URL from the sitemap tree rooted at ``root_url``.

        Encountered ``<sitemapindex>`` nodes are recursed (up to
        ``max_depth``). Encountered ``<urlset>`` nodes yield their
        ``<loc>``\\ s. Order: depth-first, in-document.

        Iteration stops after ``max_urls`` URLs have been yielded. Malformed
        XML at any level is logged and treated as an empty sitemap (the walk
        continues at sibling URLs in the parent index).
        """
        yielded = [0]  # mutable container so the helper can update it
        seen_sitemaps: set[str] = set()
        yield from self._walk(root_url, depth=0, yielded=yielded, seen=seen_sitemaps)

    # ------------------------------------------------------------- internals

    def _walk(
        self,
        url: str,
        *,
        depth: int,
        yielded: list[int],
        seen: set[str],
    ) -> Iterator[str]:
        if yielded[0] >= self._max_urls:
            return
        if url in seen:
            logger.debug("sitemap walker: skipping already-visited sitemap %s", url)
            return
        seen.add(url)

        try:
            page = self._fetcher.fetch(url)
        except Exception as exc:
            logger.warning("sitemap walker: fetch failed for %s: %r", url, exc)
            return

        body = _maybe_decompress(page.html, url=url)
        if not body.strip():
            logger.debug("sitemap walker: empty body for %s", url)
            return

        try:
            root = DefusedET.fromstring(body)
        except DefusedET.ParseError as exc:
            logger.warning("sitemap walker: XML parse failed for %s: %r", url, exc)
            return
        except Exception as exc:  # defusedxml may raise EntitiesForbidden etc.
            logger.warning("sitemap walker: XML rejected for %s: %r", url, exc)
            return

        root_tag = _strip_namespace(root.tag)
        if _SITEMAPINDEX_TAG_RE.search(root_tag):
            yield from self._walk_index(root, depth=depth, yielded=yielded, seen=seen)
        elif _URLSET_TAG_RE.search(root_tag):
            yield from self._walk_urlset(root, yielded=yielded)
        else:
            logger.debug(
                "sitemap walker: unrecognised root tag %r at %s; treating as empty",
                root_tag,
                url,
            )
            return

    def _walk_index(
        self,
        root: object,
        *,
        depth: int,
        yielded: list[int],
        seen: set[str],
    ) -> Iterator[str]:
        # ``root`` is an ElementTree Element; iterate children that are
        # ``<sitemap>`` and recurse into their ``<loc>``.
        children = list(root)  # type: ignore[call-overload]
        for child in children:
            child_tag = _strip_namespace(child.tag)
            if not _SITEMAP_TAG_RE.search(child_tag):
                continue
            loc = _first_loc(child)
            if loc is None:
                continue
            if depth + 1 > self._max_depth:
                logger.debug(
                    "sitemap walker: max_depth=%d reached, refusing to recurse into %s",
                    self._max_depth,
                    loc,
                )
                continue
            yield from self._walk(loc, depth=depth + 1, yielded=yielded, seen=seen)
            if yielded[0] >= self._max_urls:
                return

    def _walk_urlset(
        self,
        root: object,
        *,
        yielded: list[int],
    ) -> Iterator[str]:
        children = list(root)  # type: ignore[call-overload]
        for child in children:
            child_tag = _strip_namespace(child.tag)
            if not _URL_TAG_RE.search(child_tag):
                continue
            loc = _first_loc(child)
            if loc is None:
                continue
            if yielded[0] >= self._max_urls:
                return
            yielded[0] += 1
            yield loc


def _first_loc(element: object) -> str | None:
    """Return the text of the first ``<loc>`` child of ``element``, or None.

    Sitemap protocol guarantees exactly one ``<loc>`` per ``<url>`` or
    ``<sitemap>`` entry, but we tolerate quirky producers by taking the
    first match.
    """
    for child in list(element):  # type: ignore[call-overload]
        if not _LOC_TAG_RE.search(_strip_namespace(child.tag)):
            continue
        text = (child.text or "").strip()
        if text:
            return text
    return None
