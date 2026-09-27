"""Local single-user server for searching knowledge inside saved online videos."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import mimetypes
import os
import re
import sys
import threading
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm import LLMError, generate_json

STATIC_ROOT = ROOT / "static"
YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,20}$")
TIME_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2})$")
QUERY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["original_question", "english_query", "keywords", "entities", "aliases"],
    "properties": {
        "original_question": {"type": "string"},
        "english_query": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "entities": {"type": "array", "items": {"type": "string"}},
        "aliases": {"type": "array", "items": {"type": "string"}},
    },
}
QUERY_SYSTEM_PROMPT = """You transform a user's Chinese knowledge question into an English lexical retrieval query.
Return JSON only. Do not answer the question. Do not infer which video is correct. You cannot access transcripts.
Preserve proper nouns such as MCP, Claude Desktop, IDE, Registry, OAuth, REST, and SSE exactly when relevant.
english_query must be a concise English search phrase. keywords, entities, and aliases must contain short English retrieval terms.
Use the fixed keys original_question, english_query, keywords, entities, aliases and no other keys."""


class AppError(RuntimeError):
    def __init__(self, message: str, *, code: str = "bad_request", status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_within(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise AppError("无效路径。", code="path_invalid", status=404) from exc
    return resolved


def require_loopback_host(host: str) -> str:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise AppError("服务只能监听本机地址。", code="host_not_loopback")
    return host


def parse_youtube_id(value: str) -> str:
    value = str(value or "").strip()
    if YOUTUBE_ID_RE.fullmatch(value):
        return value
    parsed = urlparse(value)
    host = parsed.netloc.lower().split(":", 1)[0]
    video_id = ""
    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/", 1)[0]
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith(("/embed/", "/shorts/", "/live/")):
            parts = parsed.path.strip("/").split("/", 1)
            video_id = parts[1] if len(parts) == 2 else ""
    if not YOUTUBE_ID_RE.fullmatch(video_id):
        raise AppError("请输入有效的 YouTube 视频链接。", code="youtube_url_invalid")
    return video_id


def parse_clock(value: str) -> int:
    match = TIME_RE.fullmatch(str(value or "").strip())
    if not match:
        raise AppError("时间应写成 MM:SS 或 HH:MM:SS。", code="time_invalid")
    hours, minutes, seconds = match.groups()
    if int(minutes) > 59 or int(seconds) > 59:
        raise AppError("时间超出有效范围。", code="time_invalid")
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)


def parse_segments(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        items = value
    else:
        items = []
        for number, raw_line in enumerate(str(value or "").splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue
            match = re.match(r"^(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+)$", line)
            if not match:
                raise AppError(f"第 {number} 行无法识别，请使用“00:00-01:20 标题 | 内容”。", code="segment_invalid")
            body = match.group(3).split("|", 1)
            items.append({
                "start_seconds": parse_clock(match.group(1)),
                "end_seconds": parse_clock(match.group(2)),
                "title": body[0].strip(),
                "summary": body[-1].strip(),
                "transcript": body[-1].strip(),
            })

    result: list[dict[str, Any]] = []
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise AppError("知识片段格式无效。", code="segment_invalid")
        start = max(0, int(item.get("start_seconds") or 0))
        end = int(item.get("end_seconds") or 0)
        title = str(item.get("title") or "").strip()
        summary = str(item.get("summary") or item.get("transcript") or "").strip()
        transcript = str(item.get("transcript") or summary).strip()
        if end <= start or not title or not summary:
            raise AppError(f"第 {index} 个知识片段缺少有效时间、标题或内容。", code="segment_invalid")
        result.append({
            "segment_id": str(item.get("segment_id") or f"segment-{index:03d}"),
            "start_seconds": start,
            "end_seconds": end,
            "title": title[:160],
            "summary": summary[:1200],
            "transcript": transcript[:20000],
        })
    if not result:
        raise AppError("至少需要一个带时间的知识片段。", code="segments_missing")
    return result


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def search_tokens(value: Any) -> set[str]:
    text = normalize(value)
    text = re.sub(r"请问|我想知道|是什么|怎么样|如何|怎样|怎么|为什么|有没有|是否|的", "", text)
    tokens = {token for token in re.findall(r"[a-z0-9][a-z0-9_+.#-]*", text) if len(token) > 1}
    for chunk in re.findall(r"[\u3400-\u9fff]+", text):
        if len(chunk) <= 4:
            tokens.add(chunk)
        tokens.update(chunk[index:index + 2] for index in range(len(chunk) - 1))
    return tokens


def social_score(video: dict[str, Any]) -> float:
    return math.log1p(max(0, int(video.get("view_count") or 0))) + 0.5 * math.log1p(
        max(0, int(video.get("creator_followers") or 0))
    )


def understand_query(question: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        generation = generate_json(
            QUERY_SYSTEM_PROMPT,
            question,
            schema=QUERY_SCHEMA,
            temperature=0.0,
            max_tokens=384,
        )
    except LLMError as exc:
        raise AppError(f"DeepSeek Query Understanding 失败：{exc}", code="llm_provider_error", status=502) from exc
    value = generation.value
    if set(value) != set(QUERY_SCHEMA["required"]):
        raise AppError("DeepSeek 返回了不符合固定 schema 的结果。", code="llm_response_invalid", status=502)
    if str(value["original_question"]).strip() != str(question).strip():
        raise AppError("DeepSeek 未保留原始问题。", code="llm_response_invalid", status=502)
    cleaned = {"original_question": str(question).strip(), "english_query": str(value["english_query"]).strip()}
    if not cleaned["english_query"]:
        raise AppError("DeepSeek 未生成英文检索表达。", code="llm_response_invalid", status=502)
    for field in ("keywords", "entities", "aliases"):
        if not isinstance(value[field], list) or any(not isinstance(item, str) for item in value[field]):
            raise AppError("DeepSeek 返回了不符合固定 schema 的结果。", code="llm_response_invalid", status=502)
        cleaned[field] = list(dict.fromkeys(item.strip() for item in value[field] if item.strip()))
    return cleaned, generation.usage()


def retrieval_query(value: dict[str, Any]) -> str:
    parts = [value["english_query"], *value["keywords"], *value["entities"], *value["aliases"]]
    return " ".join(dict.fromkeys(part for part in parts if part))


def search_library(library: dict[str, Any], question: str, collection_id: str) -> list[dict[str, Any]]:
    question = normalize(question)
    query = search_tokens(question)
    if not question or not query:
        raise AppError("请描述你要解决的问题。", code="question_missing")

    collection = next(
        (item for item in library.get("collections", []) if item.get("collection_id") == collection_id),
        None,
    )
    if collection is None:
        raise AppError("收藏列表不存在。", code="collection_missing", status=404)
    allowed = set(collection.get("video_ids") or [])
    results: list[dict[str, Any]] = []

    for video in library.get("videos", []):
        if video.get("video_id") not in allowed:
            continue
        for segment in video.get("segments", []):
            title_tokens = search_tokens(segment.get("title"))
            summary_tokens = search_tokens(segment.get("summary"))
            transcript_tokens = search_tokens(segment.get("transcript"))
            matched = query & (title_tokens | summary_tokens | transcript_tokens)
            coverage = len(matched) / len(query)
            haystack = normalize(" ".join([
                str(segment.get("title") or ""),
                str(segment.get("summary") or ""),
                str(segment.get("transcript") or ""),
            ]))
            exact = question in haystack
            # ponytail: lexical matching is the MVP ceiling; replace this scorer with embeddings when real queries prove it insufficient.
            if not exact and coverage < 0.75:
                continue
            relevance = coverage * 100 + len(query & title_tokens) * 18 + len(query & summary_tokens) * 8
            if exact:
                relevance += 30
            results.append({
                "video_id": video["video_id"],
                "platform_video_id": video["platform_video_id"],
                "video_title": video["title"],
                "creator": video.get("creator", ""),
                "view_count": int(video.get("view_count") or 0),
                "creator_followers": int(video.get("creator_followers") or 0),
                "segment_id": segment["segment_id"],
                "segment_title": segment["title"],
                "summary": segment["summary"],
                "start_seconds": segment["start_seconds"],
                "end_seconds": segment["end_seconds"],
                "relevance": round(relevance, 1),
                "matched_terms": sorted(matched),
                "source_url": video["source_url"],
                "social_score": social_score(video),
            })

    results.sort(key=lambda item: (-item["relevance"], -item["social_score"], item["start_seconds"]))
    for item in results:
        item.pop("social_score", None)
    return results[:12]


class LibraryStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.library_path = self.root / "library.json"
        self.events_path = self.root / "events.jsonl"
        self.lock = threading.RLock()
        if not self.library_path.exists():
            atomic_write_json(self.library_path, {
                "schema_version": 1,
                "title": "我的视频知识库",
                "collections": [{"collection_id": "saved", "title": "已收藏", "video_ids": []}],
                "videos": [],
            })

    def read(self) -> dict[str, Any]:
        with self.lock:
            value = load_json(self.library_path, {})
        if not isinstance(value, dict):
            raise AppError("知识库文件损坏。", code="library_invalid", status=500)
        return value

    def add_video(self, body: dict[str, Any]) -> dict[str, Any]:
        source_url = str(body.get("source_url") or "").strip()
        platform_video_id = parse_youtube_id(source_url)
        collection_id = str(body.get("collection_id") or "saved").strip()
        title = str(body.get("title") or "").strip()
        if not title:
            raise AppError("请输入视频标题。", code="title_missing")
        video = {
            "video_id": f"youtube:{platform_video_id}",
            "platform": "youtube",
            "platform_video_id": platform_video_id,
            "source_url": f"https://www.youtube.com/watch?v={platform_video_id}",
            "title": title[:240],
            "creator": str(body.get("creator") or "").strip()[:160],
            "view_count": max(0, int(body.get("view_count") or 0)),
            "creator_followers": max(0, int(body.get("creator_followers") or 0)),
            "segments": parse_segments(body.get("segments") if "segments" in body else body.get("segment_text")),
            "saved_at": utc_now(),
        }
        with self.lock:
            library = self.read()
            collection = next(
                (item for item in library.get("collections", []) if item.get("collection_id") == collection_id),
                None,
            )
            if collection is None:
                raise AppError("收藏列表不存在。", code="collection_missing", status=404)
            videos = [item for item in library.get("videos", []) if item.get("video_id") != video["video_id"]]
            videos.append(video)
            library["videos"] = videos
            collection.setdefault("video_ids", [])
            if video["video_id"] not in collection["video_ids"]:
                collection["video_ids"].append(video["video_id"])
            atomic_write_json(self.library_path, library)
        return video

    def event(self, event_type: str, data: dict[str, Any]) -> None:
        allowed = {"query", "no_answer", "open_segment", "complete_segment", "resolved"}
        if event_type not in allowed:
            raise AppError("不支持的行为事件。", code="event_invalid")
        record = {
            "event_id": str(uuid.uuid4()),
            "timestamp": utc_now(),
            "type": event_type,
            "data": data,
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self.lock:
            with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())


class AppServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = os.name != "nt"

    def __init__(self, address: tuple[str, int], store: LibraryStore, static_root: Path, pipeline=None) -> None:
        self.store = store
        self.static_root = static_root.resolve()
        self.pipeline = pipeline
        super().__init__(address, RequestHandler)


class RequestHandler(BaseHTTPRequestHandler):
    server: AppServer

    def check_host(self):
        # Prevent a remote origin rebinding its hostname to this loopback service.
        host = urlparse('http://' + self.headers.get('Host', '')).hostname
        if host not in {'127.0.0.1', 'localhost', '::1'}:
            raise AppError('仅接受本机地址请求。', status=403)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def json_response(self, value: Any, status: int = 200) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def read_json(self) -> dict[str, Any]:
        # Reject browser-simple cross-origin POSTs before any paid search or write.
        # JSON requests require preflight; this loopback server does not enable CORS.
        if self.headers.get_content_type() != 'application/json':
            raise AppError('需要 JSON 请求。', status=415)
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise AppError("请求长度无效。", code="request_invalid") from exc
        if length <= 0 or length > 2_000_000:
            raise AppError("请求内容为空或过大。", code="request_invalid")
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AppError("请求不是有效 JSON。", code="request_invalid") from exc
        if not isinstance(value, dict):
            raise AppError("请求格式无效。", code="request_invalid")
        return value

    def handle_error(self, exc: Exception) -> None:
        if isinstance(exc, AppError):
            self.json_response({"ok": False, "error": exc.code, "message": str(exc)}, exc.status)
        else:
            self.json_response({"ok": False, "error": "internal_error", "message": "本地服务发生错误。"}, 500)

    def do_GET(self) -> None:
        try:
            self.check_host()
            path = urlparse(self.path).path
            if path == "/api/health":
                self.json_response({"ok": True, "mode": "mvp" if self.server.pipeline else "library"})
                return
            if path == "/api/library":
                value = self.server.store.read()
                if hasattr(self.server.pipeline, 'enqueue'):
                    value = self.server.pipeline.library(value)
                self.json_response(value)
                return
            self.serve_static(path)
        except Exception as exc:
            self.handle_error(exc)

    def do_POST(self) -> None:
        try:
            self.check_host()
            path = urlparse(self.path).path
            body = self.read_json()
            if path in ('/api/imports', '/api/obsidian'):
                import secrets
                manager = self.server.pipeline
                if not hasattr(manager, 'enqueue'):
                    raise AppError('此服务未启用自动整理，请连接章节版服务。', status=409)
                if not secrets.compare_digest(self.headers.get('X-ClipCompass-Import', ''), manager.token):
                    raise AppError('收藏连接已过期，请重新连接后重试。', status=403)
                if path == '/api/obsidian':
                    self.json_response({'ok': True, 'obsidian': manager.configure_obsidian(body)})
                    return
                self.json_response({'ok': True, 'job': manager.enqueue(body)}, HTTPStatus.ACCEPTED)
                return
            if path == "/api/videos":
                video = self.server.store.add_video(body)
                self.json_response({"ok": True, "video": video}, HTTPStatus.CREATED)
                return
            if path == "/api/search":
                library = self.server.store.read()
                if hasattr(self.server.pipeline, 'enqueue'):
                    library = self.server.pipeline.library(library)
                question = str(body.get("question") or "").strip()
                collection_id = str(body.get("collection_id") or "saved")
                if not question:
                    raise AppError("请描述你要解决的问题。", code="question_missing")
                collection = next((item for item in library.get("collections", [])
                                   if item.get("collection_id") == collection_id), None)
                if collection is None:
                    raise AppError("收藏列表不存在。", code="collection_missing", status=404)
                if self.server.pipeline:
                    try:
                        response = self.server.pipeline.search(question, set(collection.get("video_ids") or []))
                    except LLMError as exc:
                        raise AppError(f"DeepSeek Answer Gate 失败：{exc}",
                                       code="llm_provider_error", status=502) from exc
                    self.server.store.event("query" if response["answerable"] else "no_answer", {
                        "question": question[:500], "collection_id": collection_id,
                        "result_count": len(response["results"])
                    })
                    self.json_response(response)
                    return
                query_understanding, llm_usage = understand_query(question)
                results = search_library(library, retrieval_query(query_understanding), collection_id)
                self.server.store.event("query" if results else "no_answer", {
                    "question": question[:500], "collection_id": collection_id, "result_count": len(results)
                })
                self.json_response({
                    "ok": True,
                    "question": question,
                    "query_understanding": query_understanding,
                    "llm_usage": llm_usage,
                    "results": results,
                    "message": "" if results else "当前收藏列表中没有足以回答这个问题的内容。",
                })
                return
            if path == "/api/events":
                self.server.store.event(str(body.get("type") or ""), body.get("data") or {})
                self.json_response({"ok": True})
                return
            raise AppError("接口不存在。", code="not_found", status=404)
        except Exception as exc:
            self.handle_error(exc)

    def serve_static(self, request_path: str) -> None:
        relative = "index.html" if request_path == "/" else unquote(request_path.lstrip("/"))
        path = ensure_within(self.server.static_root / relative, self.server.static_root)
        if not path.is_file():
            raise AppError("页面不存在。", code="not_found", status=404)
        payload = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(payload)


def run_server(data_root: Path, *, host: str = "127.0.0.1", port: int = 0,
               open_browser: bool = True, mvp: bool = False, pipeline=None) -> int:
    require_loopback_host(host)
    if mvp and pipeline is None:
        from library.mvp_pipeline import MvpPipeline
        pipeline = MvpPipeline(understand_query)
    server = AppServer((host, port), LibraryStore(data_root), STATIC_ROOT, pipeline)
    if hasattr(pipeline, 'enqueue'):
        pipeline.start()
    url = f"http://{host}:{server.server_address[1]}"
    print(url, flush=True)
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if hasattr(pipeline, 'enqueue'):
            pipeline.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="个人收藏视频知识检索工具")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / ".video-knowledge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--mvp", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run_server(args.data_root, host=args.host, port=args.port,
                      open_browser=not args.no_open, mvp=args.mvp)


if __name__ == "__main__":
    raise SystemExit(main())
