import os
import re
from collections import Counter
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import List


TOKEN_PATTERN = re.compile(r"[a-z0-9_+-]+|[&]|[\u4e00-\u9fff]+", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
BLOCK_PATTERN = re.compile(
    r"<(h[1-3]|p|li)[^>]*>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)
TAG_PATTERN = re.compile(r"<[^>]+>")
IGNORED_BLOCK_PATTERN = re.compile(
    r"<(script|style|noscript)[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)

QUERY_EXPANSIONS = {
    "server down": ["service unavailable", "service outage", "service crash", "service timeout", "backend incident", "infrastructure incident", "kubernetes", "ingress"],
    "outage": ["unavailable", "down", "interruption", "timeout"],
    "attack": ["security incident", "intrusion", "sql injection", "ddos", "malicious traffic", "waf"],
    "model issue": ["model serving incident", "inference failure", "feature service incident", "gpu incident", "quality drop", "recommendation service"],
    "服务器挂了": ["服务不可用", "服务超时", "服务宕机", "后端故障", "基础设施故障", "kubernetes", "ingress"],
    "挂了": ["不可用", "宕机", "中断", "超时"],
    "黑客攻击": ["安全事件", "入侵", "ddos", "sql注入", "漏洞利用", "恶意流量"],
    "攻击": ["入侵", "恶意流量", "ddos", "注入"],
    "机器学习模型出问题": ["模型推理异常", "模型服务故障", "推荐质量下降", "gpu故障", "特征服务异常"],
    "模型出问题": ["模型服务故障", "推理异常", "推荐质量下降", "gpu故障"],
    "oom": ["outofmemoryerror", "memory leak", "pod restart", "内存泄漏", "内存溢出"],
}

DOMAIN_HINTS = {
    "backend": ["backend", "service", "api", "gateway", "dependency", "timeout", "circuit breaker", "后端", "服务", "接口", "超时", "降级", "熔断", "pod", "oom", "内存"],
    "sre": ["sre", "infrastructure", "kubernetes", "k8s", "ingress", "etcd", "control plane", "gateway", "cloud", "基础设施", "集群", "监控", "告警", "容量", "节点"],
    "database": ["database", "dba", "mysql", "redis", "postgresql", "replica", "connection pool", "数据库", "主从", "延迟", "慢查询", "连接池", "binlog"],
    "security": ["security", "attack", "injection", "ddos", "vulnerability", "risk", "malicious", "安全", "攻击", "入侵", "漏洞", "恶意", "waf"],
    "ai": ["ai", "algorithm", "model", "inference", "feature", "gpu", "experiment", "算法", "模型", "推理", "特征", "推荐", "质量下降", "gpu"],
    "data": ["data", "hadoop", "flink", "kafka", "hdfs", "offline", "realtime", "数据", "离线", "实时", "任务", "spark", "etl"],
    "frontend": ["frontend", "web", "cdn", "页面", "白屏", "资源加载", "兼容性", "前端"],
    "mobile": ["mobile", "app", "crash", "push", "热修复", "移动端", "崩溃率", "推送"],
}

QUERY_INTENTS = [
    {"triggers": ["server", "service", "down", "outage", "unavailable", "timeout", "服务器", "服务", "挂了", "宕机", "不可用"], "domains": ["backend", "sre"]},
    {"triggers": ["attack", "intrusion", "injection", "ddos", "malicious", "黑客", "攻击", "入侵", "漏洞", "恶意"], "domains": ["security"]},
    {"triggers": ["model", "ml", "inference", "feature", "gpu", "experiment", "机器学习", "模型", "推理", "特征", "推荐"], "domains": ["ai"]},
    {"triggers": ["database", "replica", "connection pool", "slow query", "binlog", "数据库", "主从", "延迟", "慢查询", "连接池"], "domains": ["database"]},
    {"triggers": ["hadoop", "flink", "kafka", "hdfs", "offline job", "realtime compute", "数据", "etl", "spark"], "domains": ["data"]},
    {"triggers": ["oom", "outofmemoryerror", "内存泄漏", "内存溢出"], "domains": ["backend", "sre"]},
    {"triggers": ["cdn", "白屏", "资源加载"], "domains": ["frontend"]},
]


def normalize_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", unescape(text)).strip()


def strip_tags(html_fragment: str) -> str:
    cleaned = IGNORED_BLOCK_PATTERN.sub(" ", html_fragment)
    cleaned = TAG_PATTERN.sub(" ", cleaned)
    return normalize_text(cleaned)


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def tokenize(text: str) -> List[str]:
    normalized = normalize_text(text).lower()
    tokens: List[str] = []

    for match in TOKEN_PATTERN.finditer(normalized):
        chunk = match.group(0)
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            if len(chunk) == 1:
                tokens.append(chunk)
                continue
            for size in (2, 3):
                if len(chunk) >= size:
                    tokens.extend(chunk[index:index + size] for index in range(len(chunk) - size + 1))
        else:
            tokens.append(chunk)

    return tokens


class VisibleTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._in_title = False
        self._text_parts: List[str] = []
        self._title_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"} and self._ignored_depth > 0:
            self._ignored_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignored_depth > 0:
            return

        if self._in_title:
            self._title_parts.append(data)
        self._text_parts.append(data)

    @property
    def visible_text(self) -> str:
        return normalize_text(" ".join(self._text_parts))

    @property
    def title(self) -> str:
        return normalize_text(" ".join(self._title_parts))


def extract_visible_text_and_title(html: str) -> tuple[str, str]:
    parser = VisibleTextExtractor()
    parser.feed(html)
    parser.close()
    return parser.visible_text, parser.title


def build_snippet(text: str, query: str, radius: int = 50) -> str:
    normalized_query = normalize_text(query)
    if not text:
        return ""

    for candidate in [normalized_query] + expand_query(query):
        if not candidate:
            continue
        lower_text = text.lower()
        lower_query = candidate.lower()
        position = lower_text.find(lower_query)
        if position >= 0:
            start = max(0, position - radius)
            end = min(len(text), position + len(candidate) + radius)
            snippet = text[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."
            return snippet

    if normalized_query:
        lower_text = text.lower()
        lower_query = normalized_query.lower()
        position = lower_text.find(lower_query)
        if position >= 0:
            start = max(0, position - radius)
            end = min(len(text), position + len(normalized_query) + radius)
            snippet = text[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."
            return snippet

    return text[: radius * 2] + ("..." if len(text) > radius * 2 else "")


def expand_query(query: str) -> List[str]:
    normalized = normalize_text(query).lower()
    if not normalized:
        return []

    expansions: List[str] = []
    seen = {normalized}

    for phrase, related in QUERY_EXPANSIONS.items():
        if phrase in normalized:
            for item in related:
                candidate = normalize_text(item).lower()
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    expansions.append(candidate)

    if any(token in normalized for token in ("server", "service")):
        for candidate in ("service incident", "service unavailable", "backend service", "infrastructure"):
            if candidate not in seen:
                seen.add(candidate)
                expansions.append(candidate)

    if any(token in normalized for token in ("down", "outage")):
        for candidate in ("incident", "unavailable", "interruption", "timeout"):
            if candidate not in seen:
                seen.add(candidate)
                expansions.append(candidate)

    return expansions


def build_hybrid_query(query: str) -> str:
    parts = [normalize_text(query)] + expand_query(query)
    return " ".join(part for part in parts if part)


def query_contains_trigger(combined_query: str, combined_tokens: List[str], trigger: str) -> bool:
    normalized_trigger = normalize_text(trigger).lower()
    if not normalized_trigger:
        return False
    if " " in normalized_trigger:
        return normalized_trigger in combined_query
    return normalized_trigger in combined_tokens


@dataclass
class StoredDocument:
    id: str
    title: str
    html: str
    text: str
    heading_text: str
    tokens: List[str]
    term_freq: Counter
    length: int


@dataclass
class StoredChunk:
    id: str
    doc_id: str
    doc_title: str
    heading: str
    text: str
    context_text: str
    tokens: List[str]
    term_freq: Counter
    length: int


def extract_document_chunks(html: str, title: str, doc_id: str) -> List[StoredChunk]:
    cleaned_html = IGNORED_BLOCK_PATTERN.sub(" ", html)
    headings = {"h1": title, "h2": "", "h3": ""}
    chunks: List[StoredChunk] = []
    chunk_index = 0

    for match in BLOCK_PATTERN.finditer(cleaned_html):
        tag = match.group(1).lower()
        content = strip_tags(match.group(2))
        if not content:
            continue

        if tag in {"h1", "h2", "h3"}:
            headings[tag] = content
            if tag == "h1":
                headings["h2"] = ""
                headings["h3"] = ""
            elif tag == "h2":
                headings["h3"] = ""
            continue

        heading_parts = [headings["h1"], headings["h2"], headings["h3"]]
        heading = " | ".join(part for part in heading_parts if part)
        context_text = normalize_text(f"{title} {heading} {content}")
        tokens = tokenize(context_text)
        if not tokens:
            continue

        chunks.append(
            StoredChunk(
                id=f"{doc_id}::chunk-{chunk_index}",
                doc_id=doc_id,
                doc_title=title,
                heading=heading,
                text=content,
                context_text=context_text,
                tokens=tokens,
                term_freq=Counter(tokens),
                length=len(tokens),
            )
        )
        chunk_index += 1

    if chunks:
        return chunks

    fallback_text = strip_tags(html)
    tokens = tokenize(fallback_text)
    if not tokens:
        return []
    return [
        StoredChunk(
            id=f"{doc_id}::chunk-0",
            doc_id=doc_id,
            doc_title=title,
            heading=title,
            text=fallback_text,
            context_text=normalize_text(f"{title} {fallback_text}"),
            tokens=tokens,
            term_freq=Counter(tokens),
            length=len(tokens),
        )
    ]


def extract_heading_text(html: str, title: str) -> str:
    cleaned_html = IGNORED_BLOCK_PATTERN.sub(" ", html)
    headings = [title]
    for match in re.finditer(r"<(h[1-3])[^>]*>(.*?)</\1>", cleaned_html, re.IGNORECASE | re.DOTALL):
        content = strip_tags(match.group(2))
        if content:
            headings.append(content)
    return normalize_text(" ".join(headings))
