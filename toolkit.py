"""Built-in tool registry and helpers."""

from __future__ import annotations

import os
import re
import html
from typing import Any, Dict, Mapping, Sequence, Tuple, List
from tempfile import NamedTemporaryFile

import requests
import xml.etree.ElementTree as ET
from io import StringIO
import warnings

# Optional langchain-community imports (graceful fallback when not installed)
try:
    from langchain_community.document_loaders import WikipediaLoader, ArxivLoader  # type: ignore
    _HAS_LC_COMMUNITY = True
    print("langchain_community.document_loaders.WikipediaLoader and langchain_community.document_loaders.ArxivLoader are available")
except Exception:  # pragma: no cover - optional dependency
    _HAS_LC_COMMUNITY = False
    print("langchain_community.document_loaders.WikipediaLoader and langchain_community.document_loaders.ArxivLoader are not available")

# Prefer standalone tavily integration; fall back to legacy community tool
try:
    from langchain_tavily import TavilySearch  # type: ignore
    _HAS_TAVILY_TOOL = True
except Exception:
    _HAS_TAVILY_TOOL = False
    print("warning: Tavily tool not available; please install langchain-tavily or langchain-community for web search tool")

from goob_ai.types_new import Tool, ToolRegistry

# Silence non-fatal BeautifulSoup parser guessing warnings (third-party lib behavior)
try:  # pragma: no cover - environment dependent
    from bs4 import GuessedAtParserWarning  # type: ignore
    warnings.filterwarnings("ignore", category=GuessedAtParserWarning)
except Exception:
    pass

def split_into_chunks(
    texts: Sequence[tuple[str, dict]],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[tuple[str, dict]]:
    """Split (text, metadata) pairs into chunks while preserving metadata.

    Args:
        texts: A sequence of (text, metadata) tuples.
        chunk_size: Target size of each chunk (characters).
        chunk_overlap: Overlap size between adjacent chunks (characters).

    Returns:
        A list of (chunk_text, metadata) tuples.
    """

    try:
        # Prefer robust splitter when available
        from langchain_text_splitters import (  # type: ignore
            RecursiveCharacterTextSplitter,
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " "],
        )
        out: list[tuple[str, dict]] = []
        for text, meta in texts:
            for chunk in splitter.split_text(text or ""):
                if chunk.strip():
                    safe_meta = meta if isinstance(meta, dict) else {"source": str(meta)}
                    out.append((chunk, safe_meta))
        return out
    except Exception as exc:
        print(f"warning: langchain_text_splitters not installed; using minimal fallback: {exc}")
        # Minimal fallback: simple fixed-length slicing
        out_fallback: list[tuple[str, dict]] = []
        step = max(1, chunk_size - chunk_overlap)
        for text, meta in texts:
            t = text or ""
            for i in range(0, len(t), step):
                chunk = t[i : i + chunk_size]
                if chunk.strip():
                    safe_meta = meta if isinstance(meta, dict) else {"source": str(meta)}
                    out_fallback.append((chunk, safe_meta))
        return out_fallback


def _tfidf_prerank(
    chunks: list[tuple[str, dict]],
    query: str,
    top_k: int,
) -> list[tuple[str, dict, float]]:
    """Pre-rank chunks using TF-IDF cosine similarity, with lightweight fallback.

    Args:
        chunks: List of (text, metadata) chunks.
        query: Query text to score against.
        top_k: Number of items to keep.

    Returns:
        List of (text, metadata, score) tuples sorted by score descending.
    """

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

        corpus = [c[0] for c in chunks]
        vec = TfidfVectorizer(max_features=20000)
        X_corpus = vec.fit_transform(corpus)
        X_query = vec.transform([query])
        sims = cosine_similarity(X_query, X_corpus).ravel()
        ranked = sorted(
            zip(chunks, sims),
            key=lambda x: float(x[1]),
            reverse=True,
        )[: max(1, int(top_k))]
        return [(c[0], c[1], float(s)) for c, s in ranked]
    except Exception as e:
        print(f"warning: sklearn not installed; using minimal fallback: {e}")
        # Fallback: token overlap (Jaccard-like)
        import re as _re

        q_terms = set(_re.findall(r"\w+", (query or "").lower()))
        scored: list[tuple[str, dict, float]] = []
        for text, meta in chunks:
            terms = set(_re.findall(r"\w+", (text or "").lower()))
            inter = len(q_terms & terms)
            union = len(q_terms | terms) or 1
            scored.append((text, meta, inter / union))
        scored.sort(key=lambda x: float(x[2]), reverse=True)
        return scored[: max(1, int(top_k))]


def _crossencoder_rerank(
    items: list[tuple[str, dict, float]],
    query: str,
    top_n: int,
) -> list[tuple[str, dict, float]]:
    """Optionally re-rank using a CrossEncoder; otherwise return top_n from input.

    Args:
        items: Pre-ranked (text, metadata, score) items.
        query: Query for pairwise scoring.
        top_n: Number of items to return after re-ranking.

    Returns:
        Re-ranked list limited to top_n.
    """

    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [[query, t] for (t, _m, _s) in items]
        scores = model.predict(pairs)
        merged = [
            (items[i][0], items[i][1], float(scores[i])) for i in range(len(items))
        ]
        merged.sort(key=lambda x: float(x[2]), reverse=True)
        return merged[: max(1, int(top_n))]
    except Exception as e:
        print(f"warning: sentence_transformers not installed; using minimal fallback: {e}")
        return items[: max(1, int(top_n))]


def rank_chunks(
    chunks: list[tuple[str, dict]],
    query: str,
    top_k: int = 8,
    rerank_top_n: int = 4,
) -> list[tuple[str, dict, float]]:
    """Rank chunks with TF-IDF preselection and optional CrossEncoder re-ranking.

    Args:
        chunks: List of (text, metadata) chunks.
        query: Query text for scoring.
        top_k: Number of candidates to select in pre-ranking.
        rerank_top_n: Number of final items to return after re-ranking.

    Returns:
        List of (text, metadata, score) tuples.
    """

    pre = _tfidf_prerank(chunks, query, top_k)
    return _crossencoder_rerank(pre, query, rerank_top_n)


def web_search(   
    *,
    query: str,
    max_results: int = 5,
    ) -> str:
    """Web search; args: query, max_results. Example: tool: web_search query="llm news" max_results=5
    Args:
        query: Free-text search query of keywords.
        max_results: Maximum number of results to include (maximum 5).
    """
    # normalize max_results which may come as str from tool args
    try:
        k = max(1, min(int(max_results), 10))
    except Exception:
        k = 5
    if _HAS_TAVILY_TOOL:
        try:
            tool = TavilySearch(
                tavily_api_key=os.environ.get("TAVILY_API_KEY", ""),
                max_results=k,
                include_answer=True
            )
            result = tool.invoke({"query": query})

            if isinstance(result, dict):
                items = result.get("results") or []
                answer = result.get("answer")

                blocks: list[str] = []
                for item in items[:k]:
                    title = item.get("title", "")
                    content = item.get("content")
                    score = item.get("score")
                    score_attr = f' score="{score}"' if score is not None else ""
                    blocks.append(
                        f'<Document source="tavily" title="{title}"{score_attr}/>\n{str(content or "").strip()}\n</Document>'
                    )
                summary_bits: list[str] = []
                if answer:
                    summary_bits.append(f"answer: {answer}")
                if summary_bits:
                    blocks.insert(0, "summary:\n" + "\n".join(summary_bits))
                return "\n\n---\n\n".join(blocks)

            if isinstance(result, str):
                return result
            return str(result)
        except Exception as exc:  # noqa: BLE001
            return f"error: tavily search failed: {exc}"

    # Minimal HTTP fallback
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return "error: missing TAVILY_API_KEY environment variable for Tavily web search"
    # k already computed above
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": k, "search_depth": "basic"},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("results") or []
        blocks = [
            f'<Document source="{it.get("url", "")}" title="{it.get("title", "")}"/>\n{str(it.get("content", "")).strip()}\n</Document>'
            for it in items
        ]
        return "\n\n---\n\n".join(blocks)
    except Exception as exc:  # noqa: BLE001
        return f"error: web search failed: {exc}"


def arxiv_search(*, query: str, max_results: int = 10) -> str:
    """arXiv search; args: query, max_results. Example: tool: arxiv_search query="diffusion model survey" max_results=5
    Args:
        query: Free-text search query of keywords.
        max_results: Maximum number of results to include (default 10).
    """
    # normalize max_results which may come as str
    try:
        k_loader = max(1, min(int(max_results), 10))
    except Exception:
        k_loader = 10
    try:
        import arxiv as _  # type: ignore
        if _HAS_LC_COMMUNITY:
            try:
                loader = ArxivLoader(query=query, load_max_docs=k_loader)
                # 关键：先把生成器完全消耗为列表，确保所有文档都加载完毕
                docs_list = list(loader.lazy_load())
                blocks: list[str] = []
                for d in docs_list:
                    meta = getattr(d, "metadata", {}) or {}
                    title = meta.get("Title") or meta.get("title") or ""
                    published = meta.get("Published") or meta.get("published") or ""
                    entry_id = meta.get("entry_id") or meta.get("id") or ""
                    content = getattr(d, "page_content", "") or ""
                    blocks.append(
                        f'<Document source="arXiv" title="{title}" published="{published}" id="{entry_id}"/>\n{content[:1000]}\n</Document>'
                    )
                if blocks:
                    return "\n\n---\n\n".join(blocks)
                else:
                    print("arxiv loader returned no documents, falling back to HTTP")
            except Exception as exc:  # noqa: BLE001
                print(f"warning: arxiv loader failed: {exc}, falling back to HTTP")
    except ModuleNotFoundError:
        print("warning: arxiv package not installed; Install arxiv with 'pip install -U arxiv pymupdf' for using better arxiv search.")

    # Minimal HTTP fallback
    k = k_loader
    try:
        resp = requests.get(
            "http://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": k,
                "sortBy": "relevance",
            },
            timeout=30,
        )
        resp.raise_for_status()
        response_text = resp.text
        element = ET.fromstring(response_text)
        ns = "{http://www.w3.org/2005/Atom}"
        entries = element.findall(f"{ns}entry")
        blocks: list[str] = []
        for entry in entries[:k]:
            title_el = entry.find(f"{ns}title")
            summary_el = entry.find(f"{ns}summary")
            published_el = entry.find(f"{ns}published")
            id_el = entry.find(f"{ns}id")
            title = (title_el.text or "").strip() if title_el is not None else ""
            summary_txt = (summary_el.text or "").strip() if summary_el is not None else ""
            published = (published_el.text or "").strip() if published_el is not None else ""
            arxiv_id = (id_el.text or "").strip() if id_el is not None else ""
            blocks.append(
                f'<Document source="arXiv" title="{title}" published="{published}" id="{arxiv_id}"/>\n{summary_txt}\n</Document>'
            )
        if not blocks:
            return "error: arxiv search failed: no entries returned"
        return "\n\n---\n\n".join(blocks)
    except Exception as exc:  # noqa: BLE001
        return f"error: arxiv search failed: {exc}"



def wiki_retrieve(
    *,
    query: str,
    max_results: int = 5,
    top_k: int = 8,
    rerank_top_n: int = 4,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> str:
    """Wikipedia retrieve; args: query, max_results. Example: tool: wiki_retrieve query="量子计算 概述" max_results=5

    This tool performs retrieval only and returns standardized context text for
    downstream LLM answering controlled by the agent graph.

    Args:
        query: Search query for Wikipedia.
        max_results: Max number of documents to load before chunking.
        top_k: Candidate pool size for pre-ranking.
        rerank_top_n: Number of final contexts to return.

    Returns:
        A textual payload containing ranked contexts and their sources.
    """

    # 1) Retrieve
    try:
        k_docs = max(1, min(int(max_results), 10))
    except Exception:
        k_docs = 5
    blocks: list[tuple[str, dict]] = []
    if _HAS_LC_COMMUNITY:
        docs = WikipediaLoader(query=query, load_max_docs=k_docs).lazy_load()
        for d in docs:
            content = d.page_content or ""
            meta = d.metadata or {} #include source, page, title
            safe_meta = meta if isinstance(meta, dict) else {"source": str(meta)}
            if content.strip():
                blocks.append((content, safe_meta))
            #print(blocks[0][1].keys())
    else:
        raise RuntimeError("langchain_community not available")


    # 2) Split
    chunks = split_into_chunks(blocks, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return "error: wikipedia retrieve failed: empty chunks"

    # 3) Rank
    ranked = rank_chunks(chunks, query, top_k=top_k, rerank_top_n=rerank_top_n)
    if not ranked:
        return "error: wikipedia retrieve failed: ranking produced no results"

    top_contexts = [(t, m) for (t, m, _s) in ranked]

    # 4) Format
    lines: list[str] = [f'contexts: top {len(top_contexts)} for query="{query}"']
    for i, (t, m) in enumerate(top_contexts, start=1):
        m_dict = m if isinstance(m, dict) else {}
        fallback_src = m if not isinstance(m, dict) else "Wikipedia"
        src = str(m_dict.get("source") or m_dict.get("page") or m_dict.get("title") or fallback_src or "Wikipedia").strip()
        lines.append(f"[{i}] {src}\n{t}\n---")

    return "\n\n".join(lines)


def analyze_table(*, csv_text: str) -> str:
    """Analyze CSV; args: csv_text. Example: tool: analyze_table csv_text="a,b\\n1,2"

    Args:
        csv_text: Raw CSV content as a string (including header row).

    Returns:
        A compact textual summary or an error string.
    """

    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return (
            "error: pandas not installed. Please install with 'pip install pandas'. "
            f"detail: {exc}"
        )

    try:
        df = pd.read_csv(StringIO(csv_text))
        summary = [
            f"rows={len(df)} cols={len(df.columns)}",
            f"columns: {', '.join(map(str, df.columns.tolist()))}",
            "summary:\n" + df.describe(include='all').to_string(),
        ]
        return "\n".join(summary)
    except Exception as exc:  # noqa: BLE001
        return f"error: failed to analyze table: {exc}"


def analyze_image(*, image_path: str) -> str:
    """Analyze image; args: image_path. Example: tool: analyze_image image_path=/path/to.png

    Args:
        image_path: Local path to an image file.

    Returns:
        A textual analysis including basic metadata and optional OCR text.
    """

    try:
        from PIL import Image  # type: ignore
        from pathlib import Path
        p = Path(image_path).expanduser()
        if not p.is_absolute():
            # 尝试使用当前工作目录补全相对路径
            p = (Path.cwd() / p).resolve()
        if not p.exists():
            return f"error: file not found: {p} (cwd={Path.cwd()})"
    except Exception as exc:  # noqa: BLE001
        return (
            "error: pillow (PIL) not installed. Please 'pip install pillow'. "
            f"detail: {exc}"
        )

    try:
        image = Image.open(str(p))
        print(f"analyze_image: opened image: {p} (format={image.format}, size={image.size}, mode={image.mode})")
        width, height = image.size
        mode = image.mode
        fmt = image.format
        info = [
            f"file={p}",
            f"format={fmt}",
            f"size={width}x{height}",
            f"mode={mode}",
        ]

        # Optional OCR
        try:
            import easyocr
            reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory
            result = reader.readtext(str(p),detail=0)
            print("analyze_image: OCR result:", result)
            info.append(f"ocr:{result}")
        except Exception:
            info.append(f"ocr: (easyocr not installed or OCR failed, exc: {exc})")
            # OCR optional; ignore errors
            pass
        print(info)
        return "\n".join(info)
    except Exception as exc:  # noqa: BLE001
        return f"error: failed to analyze image: {exc}"


def analyze_remote_image(*, url: str, timeout: int = 15, max_bytes: int = 5 * 1024 * 1024) -> str:
    """Analyze remote image; args: url, timeout?, max_bytes?. Example: tool: analyze_remote_image url=https://example.com/x.png timeout=15

    Args:
        url: HTTP/HTTPS URL pointing to the image resource.
        timeout: Request timeout in seconds.
        max_bytes: Maximum number of bytes to download before aborting.

    Returns:
        Analysis string produced by ``analyze_image`` or an error string.
    """

    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return f"error: unsupported URL scheme for image download: {parsed.scheme}"
    except Exception as exc:  # noqa: BLE001
        return f"error: invalid image URL: {exc}"

    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return f"error: failed to download image: {exc}"

    content_type = response.headers.get("Content-Type", "").lower()
    if content_type and "image" not in content_type:
        return f"error: URL did not return an image (content-type={content_type})"

    tmp_path: Path | None = None
    try:
        from pathlib import Path

        suffix = ""
        if "/" in content_type:
            subtype = content_type.split("/", 1)[1]
            if subtype:
                suffix = f".{subtype.split(';', 1)[0].strip()}"

        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_path = Path(tmp_file.name)
            total = 0
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    return f"error: remote image exceeds size limit of {max_bytes} bytes"
                tmp_file.write(chunk)

        analysis = analyze_image(image_path=str(tmp_path))
        if analysis.startswith("error"):
            return analysis
        return f"source_url={url}\n{analysis}"
    except Exception as exc:  # noqa: BLE001
        return f"error: failed to process remote image: {exc}"
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass


def analyze_video(*, url: str) -> str:
    """Analyze video; args: url. Example: tool: analyze_video url=https://www.youtube.com/watch?v=XXXX

    Args:
        url: YouTube video URL.

    Returns:
        A textual summary including title, uploader, duration, description, and chapters list.
        On error returns an error string starting with "error:".
    """

    try:
        import yt_dlp  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return (
            "error: yt_dlp not installed. Please 'pip install yt-dlp'. "
            f"detail: {exc}"
        )

    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'no_playlist': True,
            'youtube_include_dash_manifest': False,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False, process=False)
        if not info:
            return "error: could not extract video information"
        title = info.get('title', 'Unknown')
        description = info.get('description', '') or ''
        uploader = info.get('uploader', '') or ''
        duration = info.get('duration')
        duration_s = f"{duration}s" if duration else "unknown"
        lines = [
            f"title={title}",
            f"uploader={uploader}",
            f"duration={duration_s}",
            f"description:\n{description[:1200]}" if description else "description: (none)",
        ]
        
        # Helper functions for chapter formatting and title cleaning
        def _fmt_ch(s: int | None) -> str:
            if s is None:
                return "end"
            h, r = divmod(max(0, s), 3600)
            m, s2 = divmod(r, 60)
            return f"{h:02d}:{m:02d}:{s2:02d}"
        def _clean_title(txt: str) -> str:
            # Remove leading/trailing timestamps like "00:00" / "1:23:45" and adjacent separators
            import re as _re
            s = txt.strip()
            s = _re.sub(r"^(?:\d{1,2}:)?\d{1,2}:\d{2}\s*[-–—:]*\s*", "", s)
            s = _re.sub(r"\s*[-–—:]*\s*(?:\d{1,2}:)?\d{1,2}:\d{2}$", "", s)
            return s.strip()
        
        # Append chapters if available (from metadata or description)
        chapters = info.get('chapters') or []
        duration_total = int(info.get('duration') or 0)
        if chapters:
            lines.append("\nchapters (from metadata):")
            for idx, ch in enumerate(chapters, start=1):
                ch_title = _clean_title(str(ch.get('title', '') or ''))
                st = int(ch.get('start_time') or 0)
                # Prefer provided end_time; otherwise use next chapter start; fallback to duration
                raw_end = ch.get('end_time')
                if raw_end is not None:
                    en = int(raw_end)
                else:
                    if idx < len(chapters):
                        en = int(chapters[idx].get('start_time') or 0)
                    else:
                        en = duration_total if duration_total > 0 else None
                lines.append(f"  {idx}. [{_fmt_ch(st)} - {_fmt_ch(en)}] {_clean_title(ch_title)}")
        else:
            # Fallback: parse description timestamps
            import re as _re
            ts_re = _re.compile(r"(?P<ts>(?:\d{1,2}:)?\d{1,2}:\d{2})(?:\s+|-|–|—|:)\s*(?P<title>.+)")
            found: list[tuple[int, str]] = []
            def _to_sec(t: str) -> int:
                p = [int(x) for x in t.split(":")]
                if len(p) == 3:
                    return p[0] * 3600 + p[1] * 60 + p[2]
                if len(p) == 2:
                    return p[0] * 60 + p[1]
                return 0
            for line in (description or '').splitlines():
                m = ts_re.search(line.strip())
                if m:
                    found.append((_to_sec(m.group('ts')), m.group('title').strip()))
            if found:
                found.sort(key=lambda x: x[0])
                lines.append("\nchapters (from description):")
                for idx, (sec, ch_title) in enumerate(found, start=1):
                    next_sec = found[idx][0] if idx < len(found) else None
                    def _fmt_ch(s: int) -> str:
                        h, r = divmod(max(0, s), 3600)
                        m, s2 = divmod(r, 60)
                        return f"{h:02d}:{m:02d}:{s2:02d}"
                    rng = f"[{_fmt_ch(sec)} - {_fmt_ch(next_sec)}]" if next_sec else f"[{_fmt_ch(sec)} - end]"
                    lines.append(f"  {idx}. {rng} {_clean_title(ch_title)}")

    except Exception as exc:  # noqa: BLE001
        return f"error: failed to analyze video metadata: {exc}"

    return "\n".join(lines)


def analyze_video_by_chapter(*, url: str, start: str | int, end: str | int | None = None, subtitle_langs: list[str] | None = None) -> str:
    """Extract subtitles; args: url, start, end?, subtitle_langs?. Example: tool: analyze_video_by_chapter url=YTB start=00:00 end=05:00

    Args:
        url: YouTube video URL.
        start: Start time in seconds or as "HH:MM:SS" / "MM:SS" (inclusive).
        end: Optional end time in seconds or as "HH:MM:SS" / "MM:SS" (inclusive). If None, reads until the end.
        subtitle_langs: Preferred subtitle languages ordered by priority.

    Returns:
        Subtitles text for the specified time range (and metadata about language/range), or an error string.
    """

    def _parse_ts_to_seconds(ts: str | int | None) -> int | None:
        if ts is None:
            return None
        if isinstance(ts, int):
            return ts if ts >= 0 else 0
        t = ts.strip()
        if not t:
            return 0
        parts = t.split(":")
        try:
            if len(parts) == 3:
                h, m, s = (int(parts[0]), int(parts[1]), int(parts[2]))
                return max(0, h * 3600 + m * 60 + s)
            if len(parts) == 2:
                m, s = (int(parts[0]), int(parts[1]))
                return max(0, m * 60 + s)
            # Fallback: plain seconds string
            return max(0, int(t))
        except Exception:
            return 0

    start_s = _parse_ts_to_seconds(start)
    end_s = _parse_ts_to_seconds(end)

    if start_s is None or start_s < 0:
        return "error: invalid start; must be non-negative seconds or timestamp"
    if end_s is not None and end_s < start_s:
        return "error: invalid end; must be >= start"

    try:
        import yt_dlp  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return (
            "error: yt_dlp not installed. Please 'pip install yt-dlp'. "
            f"detail: {exc}"
        )

    def _fmt(ts: int | None) -> str:
        if ts is None:
            return "end"
        h, r = divmod(max(0, int(ts)), 3600)
        m, s2 = divmod(r, 60)
        return f"{h:02d}:{m:02d}:{s2:02d}"

    header = f"range: [{_fmt(start_s)} - {_fmt(end_s)}]"
    lines: list[str] = [header]

    langs = subtitle_langs or ["en", "en-US", "en-GB"]
    try:
        import tempfile
        import os as _os

        with tempfile.TemporaryDirectory() as tmpdir:
            sub_opts = {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'skip_download': True,
                'subtitleslangs': langs,
                'quiet': True,
                'no_warnings': True,
                'no_playlist': True,
                'outtmpl': _os.path.join(tmpdir, '%(title)s.%(ext)s'),
                'subtitlesformat': 'vtt/srt/best',
            }
            with yt_dlp.YoutubeDL(sub_opts) as ydl_dl:
                info2 = ydl_dl.extract_info(url, download=False)
                subs = (info2 or {}).get('subtitles', {}) or {}
                autos = (info2 or {}).get('automatic_captions', {}) or {}

                merged = {**autos, **subs}
                selected_lang: str | None = None
                for lang in langs:
                    if lang in merged:
                        selected_lang = lang
                        break

                if not selected_lang:
                    lines.append("subtitles: (none)")
                    return "\n".join(lines)

                sub_opts_specific = dict(sub_opts)
                sub_opts_specific['subtitleslangs'] = [selected_lang]
                with yt_dlp.YoutubeDL(sub_opts_specific) as ydl_dl_lang:
                    ydl_dl_lang.extract_info(url, download=True)

                sub_text: str | None = None
                selected_file: str | None = None
                for fname in _os.listdir(tmpdir):
                    if fname.endswith(f".{selected_lang}.vtt") or fname.endswith(f".{selected_lang}.srt"):
                        with open(_os.path.join(tmpdir, fname), 'r', encoding='utf-8', errors='ignore') as f:
                            sub_text = f.read()
                        selected_file = fname
                        break

                if sub_text and sub_text.strip():
                    lines.append(f"subtitles_language={selected_lang}")
                    try:
                        is_vtt = (selected_file or '').lower().endswith('.vtt')
                        def _parse_ts_to_ms(ts: str) -> int:
                            t = ts.replace(',', '.')
                            parts = t.split(':')
                            if len(parts) == 3:
                                h, m, s = parts
                            elif len(parts) == 2:
                                h, m, s = '0', parts[0], parts[1]
                            else:
                                return 0
                            sec, ms = (s.split('.') + ['0'])[:2]
                            try:
                                return int(h) * 3600000 + int(m) * 60000 + int(sec) * 1000 + int(ms.ljust(3, '0')[:3])
                            except Exception:
                                return 0
                        def _filter_subs(text: str) -> str:
                            time_arrow = '-->'
                            out_lines: list[str] = []
                            buf: list[str] = []
                            cue_start_ms = 0
                            cue_end_ms = 0
                            start_ms = max(0, int(start_s) * 1000)
                            end_ms = None if end_s is None else max(0, int(end_s) * 1000)
                            lines_in = text.splitlines()
                            i = 0
                            while i < len(lines_in):
                                line = lines_in[i]
                                if line.strip().isdigit() and i + 1 < len(lines_in) and time_arrow in lines_in[i + 1]:
                                    i += 1
                                    line = lines_in[i]
                                if time_arrow in line:
                                    if buf:
                                        if (cue_end_ms >= start_ms) and (end_ms is None or cue_start_ms <= end_ms):
                                            out_lines.extend(buf)
                                            out_lines.append('')
                                        buf = []
                                    parts = [p.strip() for p in line.split(time_arrow, 1)]
                                    if len(parts) == 2:
                                        cue_start_ms = _parse_ts_to_ms(parts[0])
                                        end_part = parts[1].split(' ', 1)[0]
                                        cue_end_ms = _parse_ts_to_ms(end_part)
                                    buf.append(line)
                                    i += 1
                                    while i < len(lines_in) and lines_in[i].strip() != '':
                                        buf.append(lines_in[i])
                                        i += 1
                                    if (cue_end_ms >= start_ms) and (end_ms is None or cue_start_ms <= end_ms):
                                        out_lines.extend(buf)
                                        out_lines.append('')
                                    buf = []
                                else:
                                    i += 1
                            if buf:
                                if (cue_end_ms >= start_ms) and (end_ms is None or cue_start_ms <= end_ms):
                                    out_lines.extend(buf)
                            if is_vtt:
                                return 'WEBVTT\n\n' + '\n'.join(out_lines).strip()
                            return '\n'.join(out_lines).strip()
                        sub_text = _filter_subs(sub_text)
                        lines.append(f"subtitles_range=[{_fmt(start_s)} - {_fmt(end_s)}]")
                    except Exception as _exc:
                        lines.append(f"subtitles_range_error: {_exc}")
                    lines.append("subtitles:\n" + sub_text[:8000])
                else:
                    lines.append("subtitles: (downloaded but empty)")

        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        lines.append(f"subtitles_error: {exc}")
        return "\n".join(lines)

def calculator(*, expression: str) -> str:
    """Calculator; args: expression. Example: tool: calculator expression=2*(3+4)-5/2

    Args:
        expression: Expression string like "2*(3+4)-5/2".

    Returns:
        Result string like "result=..." or an error string starting with "error:".
    """
    import ast
    import math

    bin_ops: dict[type[ast.AST], callable] = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b,
    }
    unary_ops: dict[type[ast.AST], callable] = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("only numeric constants are allowed")
        if hasattr(ast, "Num") and isinstance(node, getattr(ast, "Num")):  # type: ignore[attr-defined]
            return float(getattr(node, "n"))  # type: ignore[attr-defined]
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary_ops:
            return float(unary_ops[type(node.op)](_eval(node.operand)))
        if isinstance(node, ast.BinOp) and type(node.op) in bin_ops:
            left = _eval(node.left)
            right = _eval(node.right)
            return float(bin_ops[type(node.op)](left, right))
        raise ValueError(f"unsupported node: {type(node).__name__}")

    try:
        src = (expression or "").strip()
        if not src:
            return "error: empty expression"
        parsed = ast.parse(src, mode="eval")
        result = _eval(parsed)
        if math.isfinite(result) and abs(result - int(result)) < 1e-12:
            return f"result={int(result)}"
        return f"result={result}"
    except ZeroDivisionError:
        return "error: division by zero"
    except Exception as exc:  # noqa: BLE001
        return f"error: calculator failed: {exc}"

def default_tools() -> ToolRegistry:
    """Return the default tool registry.

    Returns:
        A mapping of tool names to callable tools.
    """

    registry: Mapping[str, Tool] = {
        "wiki_retrieve": wiki_retrieve,
        "calculator": calculator,
        "web_search": web_search,
        "arxiv_search": arxiv_search,
        "analyze_table": analyze_table,
        "analyze_image": analyze_image,
        "analyze_remote_image": analyze_remote_image,
        "analyze_video_by_chapter": analyze_video_by_chapter,
        "analyze_video": analyze_video,
    }
    return registry


