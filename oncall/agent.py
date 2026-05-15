import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterator, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from .schemas import ChatHistoryItem, ReadFileArgs
from .search import DocumentStore
from .semantic import SemanticSearchEngine
from .text import (
    DOMAIN_HINTS,
    StoredDocument,
    build_hybrid_query,
    build_snippet,
    expand_query,
    extract_document_chunks,
    normalize_text,
    tokenize,
)


def encode_sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@dataclass
class ToolExecution:
    output: str
    events: List[dict]
    message: ToolMessage


class DashScopeChatClient:
    def has_api_key(self) -> bool:
        return bool(os.getenv("DASHSCOPE_API_KEY", "").strip())

    def _api_key(self) -> str:
        api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing DASHSCOPE_API_KEY. Set it before using /v3 chat.")
        return api_key

    def _base_url(self) -> str:
        return os.getenv("DASHSCOPE_CHAT_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")

    def _model_name(self) -> str:
        return os.getenv("DASHSCOPE_CHAT_MODEL", "qwen3.6-plus").strip() or "qwen3.6-plus"

    def _request(self, payload: dict, stream: bool = False) -> Request:
        if stream:
            payload = {**payload, "stream": True}
        return Request(
            f"{self._base_url()}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

    def complete(self, messages: List[dict], tools: List[dict]) -> dict:
        payload = {
            "model": self._model_name(),
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.2,
        }
        request = self._request(payload)
        try:
            with urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"DashScope chat request failed: HTTP {exc.code}. {details}".strip()) from exc
        except URLError as exc:
            raise RuntimeError(f"DashScope chat request failed: {exc.reason}") from exc

    def stream(self, messages: List[dict], tools: List[dict]) -> Iterator[dict]:
        payload = {
            "model": self._model_name(),
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.2,
        }
        request = self._request(payload, stream=True)
        try:
            with urlopen(request, timeout=90) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    yield json.loads(data)
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"DashScope chat request failed: HTTP {exc.code}. {details}".strip()) from exc
        except URLError as exc:
            raise RuntimeError(f"DashScope chat request failed: {exc.reason}") from exc


class ReadFileToolRuntime:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._cache: Dict[str, tuple[int, str]] = {}
        self.tool = StructuredTool.from_function(
            func=self.read_file,
            name="readFile",
            description="Read one exact file from the data directory, such as sop-001.html.",
            args_schema=ReadFileArgs,
        )

    def openai_spec(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.tool.name,
                "description": self.tool.description,
                "parameters": ReadFileArgs.model_json_schema(),
            },
        }

    def read_file(self, fname: str) -> str:
        normalized = fname.strip().replace("\\", "/")
        if not normalized:
            raise RuntimeError("fname is required")
        if "*" in normalized or "?" in normalized:
            raise RuntimeError("Wildcards are not allowed")
        candidate = (self._data_dir / normalized).resolve()
        data_root = self._data_dir.resolve()
        if data_root not in candidate.parents or not candidate.is_file():
            raise RuntimeError(f"File not found or access denied: {fname}")

        stat = candidate.stat()
        cache_key = str(candidate)
        cached = self._cache.get(cache_key)
        if cached and cached[0] == stat.st_mtime_ns:
            return cached[1]

        content = candidate.read_text(encoding="utf-8")
        self._cache[cache_key] = (stat.st_mtime_ns, content)
        if len(self._cache) > 128:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        return content

    def execute(self, tool_call: dict) -> ToolExecution:
        function_payload = tool_call.get("function", {})
        name = function_payload.get("name", "")
        raw_arguments = function_payload.get("arguments", "{}")
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            arguments = {"fname": ""}

        fname = str(arguments.get("fname", "")).strip()
        events = [
            {
                "type": "tool_call",
                "name": name,
                "fname": fname,
                "status": "started",
                "message": f"正在读取 {fname}" if fname else "正在读取文件",
            }
        ]

        if name != self.tool.name:
            output = f"Unsupported tool: {name}"
            events.append(
                {
                    "type": "tool_call",
                    "name": name,
                    "fname": fname,
                    "status": "failed",
                    "message": f"读取 {fname or '文件'} 失败：不支持的工具",
                }
            )
        else:
            try:
                output = self.tool.invoke({"fname": fname})
                events.append(
                    {
                        "type": "tool_call",
                        "name": name,
                        "fname": fname,
                        "status": "completed",
                        "message": f"已读取 {fname}",
                    }
                )
            except Exception as exc:
                output = str(exc)
                events.append(
                    {
                        "type": "tool_call",
                        "name": name,
                        "fname": fname,
                        "status": "failed",
                        "message": f"读取 {fname or '文件'} 失败：{exc}",
                    }
                )

        return ToolExecution(
            output=output,
            events=events,
            message=ToolMessage(content=output, tool_call_id=tool_call.get("id", "")),
        )


class OnCallChatAgent:
    CROSS_TEAM_DOC_IDS = ("sop-001", "sop-002", "sop-004", "sop-005")
    CROSS_TEAM_KEYWORDS = (
        "p0",
        "p1",
        "故障响应",
        "响应流程",
        "升级流程",
        "升级路径",
        "故障分级",
        "跨团队",
        "应急响应",
        "统一流程",
        "on-call流程",
        "值班流程",
    )
    QUERY_TOPIC_DOCS = {
        "security": ("sop-005", "sop-004", "sop-001"),
        "database": ("sop-002", "sop-004", "sop-001"),
        "backend": ("sop-001", "sop-004", "sop-002"),
        "sre": ("sop-004", "sop-001", "sop-002"),
        "ai": ("sop-008", "sop-004", "sop-001"),
        "data": ("sop-006", "sop-004", "sop-002"),
    }

    def __init__(self, data_dir: Path, document_store: DocumentStore, semantic_engine: SemanticSearchEngine) -> None:
        self._lock = RLock()
        self._client = DashScopeChatClient()
        self._store = document_store
        self._semantic_engine = semantic_engine
        self._read_file_tool = ReadFileToolRuntime(data_dir)
        self._retrieval_cache: Dict[str, List[dict]] = {}

    def _tool_spec(self) -> List[dict]:
        return [self._read_file_tool.openai_spec()]

    def _has_api_key(self) -> bool:
        return self._client.has_api_key()

    def _cache_retrieval(self, key: str, candidates: List[dict]) -> List[dict]:
        self._retrieval_cache[key] = [dict(item) for item in candidates]
        if len(self._retrieval_cache) > 128:
            oldest_key = next(iter(self._retrieval_cache))
            del self._retrieval_cache[oldest_key]
        return [dict(item) for item in candidates]

    def _document_by_id(self, doc_id: str) -> Optional[StoredDocument]:
        for document in self._store.list_documents():
            if document.id == doc_id:
                return document
        return None

    def _is_cross_team_query(self, message: str, history: List[ChatHistoryItem]) -> bool:
        text_parts = [message]
        for item in history[-4:]:
            if item.role.strip().lower() == "user":
                text_parts.append(item.content)
        normalized = normalize_text(" ".join(text_parts)).lower()
        return any(keyword in normalized for keyword in self.CROSS_TEAM_KEYWORDS)

    def _topic_priority_doc_ids(self, message: str, history: List[ChatHistoryItem]) -> List[str]:
        text_parts = [message]
        text_parts.extend(item.content for item in history[-4:] if item.role.strip().lower() == "user")
        normalized = normalize_text(" ".join(text_parts)).lower()
        priorities: List[str] = []

        for domain, hints in DOMAIN_HINTS.items():
            if any(hint.lower() in normalized for hint in hints):
                for doc_id in self.QUERY_TOPIC_DOCS.get(domain, ()):
                    if doc_id not in priorities:
                        priorities.append(doc_id)
        return priorities

    def _promote_topic_candidates(
        self,
        merged: Dict[str, dict],
        message: str,
        history: List[ChatHistoryItem],
        base_score: float = 1.15,
    ) -> None:
        query = normalize_text(message)
        for rank, doc_id in enumerate(self._topic_priority_doc_ids(message, history)):
            boost_score = max(0.7, base_score - (rank * 0.12))
            if doc_id in merged:
                merged[doc_id]["score"] += boost_score
                continue

            document = self._document_by_id(doc_id)
            if not document:
                continue
            merged[doc_id] = {
                "id": document.id,
                "title": document.title,
                "snippet": build_snippet(document.text, query) if query else "",
                "score": boost_score,
                "source": "topic-router",
            }

    def _merge_candidate(self, merged: Dict[str, dict], result: dict, source: str) -> None:
        existing = merged.get(result["id"])
        score = float(result.get("score", 0.0))
        payload = {
            "id": result["id"],
            "title": result["title"],
            "snippet": result.get("snippet", ""),
            "score": score,
            "source": source,
        }
        if existing is None or score > existing["score"]:
            merged[result["id"]] = payload

    def _promote_cross_team_candidates(self, merged: Dict[str, dict], message: str, history: List[ChatHistoryItem]) -> None:
        query = normalize_text(message)
        priority_doc_ids = self._topic_priority_doc_ids(message, history) + list(self.CROSS_TEAM_DOC_IDS)
        for rank, doc_id in enumerate(priority_doc_ids):
            if doc_id in merged:
                merged[doc_id]["score"] += max(0.45, 0.85 - (rank * 0.1))
                continue
            document = self._document_by_id(doc_id)
            if not document:
                continue
            snippet = build_snippet(document.text, query) if query else ""
            merged[doc_id] = {
                "id": document.id,
                "title": document.title,
                "snippet": snippet,
                "score": max(0.45, 0.85 - (rank * 0.1)),
                "source": "cross-team",
            }

    def _select_documents_to_read(self, candidates: List[dict], message: str, history: List[ChatHistoryItem]) -> List[dict]:
        if not candidates:
            fallback = [document for document in self._store.list_documents() if document.id.startswith("sop-")]
            if not fallback:
                return []
            if self._is_cross_team_query(message, history):
                preferred = [doc for doc in fallback if doc.id in self.CROSS_TEAM_DOC_IDS]
                return [
                    {"id": document.id, "title": document.title, "snippet": "", "score": 0.0, "source": "fallback"}
                    for document in preferred[:3]
                ]
            document = fallback[0]
            return [{"id": document.id, "title": document.title, "snippet": "", "score": 0.0, "source": "fallback"}]

        if not self._is_cross_team_query(message, history):
            return candidates[:1]

        selected: List[dict] = []
        seen_ids = set()
        priority_doc_ids = self._topic_priority_doc_ids(message, history) + list(self.CROSS_TEAM_DOC_IDS)

        for doc_id in priority_doc_ids:
            candidate = next((item for item in candidates if item["id"] == doc_id), None)
            if candidate and candidate["id"] not in seen_ids:
                selected.append(candidate)
                seen_ids.add(candidate["id"])
            if len(selected) >= 3:
                break

        for candidate in candidates:
            if candidate["id"] in seen_ids:
                continue
            selected.append(candidate)
            seen_ids.add(candidate["id"])
            if len(selected) >= 3:
                break

        return selected[:3]

    def _rag_candidates(self, message: str, history: List[ChatHistoryItem], limit: int = 4) -> List[dict]:
        query_parts: List[str] = []
        for item in history[-4:]:
            if item.role.strip().lower() == "user":
                query_parts.append(item.content)
        query_parts.append(message)
        query = normalize_text(" ".join(part for part in query_parts if part))
        if not query:
            return []

        cache_key = f"{limit}:{query.lower()}"
        cached = self._retrieval_cache.get(cache_key)
        if cached is not None:
            return [dict(item) for item in cached]

        merged: Dict[str, dict] = {}
        keyword_results = self._store.search(query, limit=limit)
        for result in keyword_results:
            self._merge_candidate(merged, result, "keyword")

        should_run_semantic = True
        if keyword_results:
            top_keyword_score = float(keyword_results[0].get("score", 0.0))
            if top_keyword_score >= 2.0 and len(keyword_results) >= min(3, limit):
                should_run_semantic = False

        if should_run_semantic:
            try:
                for result in self._semantic_engine.search(query, limit=limit):
                    self._merge_candidate(merged, result, "semantic")
            except RuntimeError:
                pass

        self._promote_topic_candidates(merged, message, history)

        if self._is_cross_team_query(message, history):
            self._promote_cross_team_candidates(merged, message, history)

        candidates = sorted(merged.values(), key=lambda item: item["score"], reverse=True)
        if candidates:
            return self._cache_retrieval(cache_key, candidates[:limit])

        fallback_candidates = [
            {
                "id": document.id,
                "title": document.title,
                "snippet": "",
                "score": 0.0,
                "source": "fallback",
            }
            for document in self._store.list_documents()[: min(limit, 4)]
        ]
        return self._cache_retrieval(cache_key, fallback_candidates)

    def _candidate_files_text(self, candidates: List[dict]) -> str:
        if not candidates:
            return "No candidate files are available."

        lines = []
        for index, candidate in enumerate(candidates, start=1):
            snippet = normalize_text(candidate.get("snippet", ""))
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            lines.append(f"{index}. {candidate['id']}.html - {candidate['title']}")
            if snippet:
                lines.append(f"   snippet: {snippet}")
        return "\n".join(lines)

    def _system_prompt(self, candidates: List[dict]) -> str:
        prompt = (
            "You are an On-Call assistant.\n"
            "Answer only from SOP documents in data/.\n"
            "Your only tool is readFile(fname), which reads one exact file.\n"
            "Call readFile before relying on document details. Never claim to have read a file unless you actually called the tool.\n"
            "Do not list directories, do not use wildcards, and do not read outside data/.\n"
            "The system may provide retrieved candidate files. Prefer those candidates first when they match the user request.\n"
            "Prefer reading the 1-3 most relevant files, then answer with concise, actionable troubleshooting steps.\n"
            "If the question is vague, choose the closest SOP, explain the assumption briefly, and keep moving.\n"
            "When structure helps, use clean Markdown: headings, numbered steps, bullets, blockquotes, and `inline code`.\n"
            "If the SOP does not support a claim, say that clearly instead of guessing.\n"
            "Retrieved candidate files:\n"
            f"{self._candidate_files_text(candidates)}"
        )
        candidate_ids = {candidate["id"] for candidate in candidates}
        if candidate_ids.intersection(self.CROSS_TEAM_DOC_IDS):
            prompt += (
                "\nIf the user asks for a company-wide incident flow, P0/P1 response, escalation flow, or cross-team process, "
                "you must read multiple SOPs before answering and synthesize a unified response."
            )
        return prompt

    def _build_lc_messages(self, history: List[ChatHistoryItem], message: str, candidates: List[dict]) -> List[Any]:
        messages: List[Any] = [SystemMessage(content=self._system_prompt(candidates))]
        for item in history:
            role = item.role.strip().lower()
            if role == "user":
                messages.append(HumanMessage(content=item.content))
            elif role == "assistant":
                messages.append(AIMessage(content=item.content))
        messages.append(HumanMessage(content=message))
        return messages

    def _serialize_lc_message(self, message: Any) -> dict:
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}
        if isinstance(message, ToolMessage):
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }
        if isinstance(message, AIMessage):
            payload = {"role": "assistant", "content": message.content}
            tool_calls = message.additional_kwargs.get("tool_calls")
            if tool_calls:
                payload["tool_calls"] = tool_calls
            return payload
        raise TypeError(f"Unsupported message type: {type(message)!r}")

    def _serialize_lc_messages(self, messages: List[Any]) -> List[dict]:
        return [self._serialize_lc_message(message) for message in messages]

    def _append_assistant_tool_calls(self, messages: List[Any], content: str, tool_calls: List[dict]) -> None:
        messages.append(AIMessage(content=content, additional_kwargs={"tool_calls": tool_calls}))

    def _execute_tool_call(self, tool_call: dict) -> ToolExecution:
        return self._read_file_tool.execute(tool_call)

    def _read_documents_for_offline_answer(self, message: str, history: List[ChatHistoryItem]) -> tuple[List[dict], List[dict]]:
        candidates = self._rag_candidates(message, history)
        selected = self._select_documents_to_read(candidates, message, history)

        tool_logs: List[dict] = []
        loaded_documents: List[dict] = []
        for index, candidate in enumerate(selected):
            tool_call = {
                "id": f"offline-read-{index}",
                "type": "function",
                "function": {
                    "name": self._read_file_tool.tool.name,
                    "arguments": json.dumps({"fname": f"{candidate['id']}.html"}, ensure_ascii=False),
                },
            }
            execution = self._execute_tool_call(tool_call)
            tool_logs.extend(execution.events)
            loaded_documents.append(
                {
                    "id": candidate["id"],
                    "title": candidate["title"],
                    "html": execution.output,
                }
            )
        return loaded_documents, tool_logs

    def _rank_chunk_texts(self, message: str, document_id: str, title: str, html: str, limit: int = 3) -> List[str]:
        chunks = extract_document_chunks(html, title, document_id)
        if not chunks:
            return []

        query = build_hybrid_query(message)
        query_tokens = tokenize(query)
        normalized_query = normalize_text(message).lower()
        expanded_terms = [term.lower() for term in expand_query(message)]
        ranked: List[tuple[float, str]] = []
        for chunk in chunks:
            heading_text = normalize_text(f"{chunk.doc_title} {chunk.heading}").lower()
            body_text = chunk.context_text.lower()
            token_overlap = sum(chunk.term_freq.get(token, 0) for token in query_tokens)
            phrase_hits = sum(1 for phrase in [normalized_query] + expanded_terms if phrase and phrase in body_text)
            heading_hits = sum(1 for token in query_tokens if token and token in heading_text)
            score = (1.4 * token_overlap) + (1.2 * phrase_hits) + (0.8 * heading_hits)
            ranked.append((score, chunk.text))

        ranked.sort(key=lambda item: item[0], reverse=True)
        seen = set()
        results: List[str] = []
        for _, text in ranked:
            summary = self._shorten_guidance(text)
            if not summary or summary in seen:
                continue
            seen.add(summary)
            results.append(summary)
            if len(results) >= limit:
                break
        return results

    def _shorten_guidance(self, text: str, limit: int = 150) -> str:
        normalized = normalize_text(text)
        if len(normalized) <= limit:
            return normalized

        parts = [part.strip() for part in re.split(r"[。！？；]", normalized) if part.strip()]
        snippet = ""
        for part in parts:
            candidate = f"{snippet}；{part}" if snippet else part
            if len(candidate) > limit:
                break
            snippet = candidate
        return snippet or (normalized[: limit - 3] + "...")

    def _is_useful_guidance(self, text: str) -> bool:
        normalized = normalize_text(text)
        if not normalized:
            return False
        metadata_prefixes = ("文档编号", "版本", "最后更新", "适用范围", "负责人", "团队职责")
        return not normalized.startswith(metadata_prefixes)

    def _build_offline_reply(self, message: str, documents: List[dict]) -> str:
        if not documents:
            return "我暂时没有找到可参考的 SOP 文档。"

        normalized_message = normalize_text(message).lower()
        sections: List[str] = []
        if len(documents) == 1:
            document = documents[0]
            sections.append(f"## 处理建议\n\n已参考 `{document['id']}.html`（{document['title']}）。")
            guidance = self._rank_chunk_texts(message, document["id"], document["title"], document["html"])
            guidance = [item for item in guidance if self._is_useful_guidance(item)]
            if guidance:
                sections.extend(f"{index}. {item}" for index, item in enumerate(guidance, start=1))
            else:
                sections.append("1. 先核对告警范围、影响面和最近变更。")
                sections.append("2. 再根据 SOP 中对应场景执行排查与止损。")
        else:
            sections.append("## 综合处理建议\n")
            sections.append("已综合参考以下 SOP：")
            for document in documents:
                sections.append(f"- `{document['id']}.html`：{document['title']}")
            sections.append("")
            if self._is_cross_team_query(message, []):
                sections.append("### 建议的统一流程")
                sections.append("1. 先确认故障级别、影响范围、持续时间，以及是否命中核心业务链路。")
                sections.append("2. 立即拉起对应值班群，同步当前现象、已知影响、最近变更和下一次更新时间。")
                sections.append("3. 按团队分工并行排查：应用与依赖、数据库、基础设施、安全风险。")
                sections.append("4. 若已达到 P0/P1 标准，立即按各团队 SOP 的升级路径升级到负责人。")
                sections.append("5. 在止血后持续更新状态，确认恢复标准、回滚方案和后续复盘责任人。")
                sections.append("")
            for document in documents:
                guidance = self._rank_chunk_texts(message, document["id"], document["title"], document["html"], limit=2)
                guidance = [item for item in guidance if self._is_useful_guidance(item)]
                sections.append(f"### `{document['id']}.html` - {document['title']}")
                if guidance:
                    sections.extend(f"- {item}" for item in guidance)

        if any(term in normalized_message for term in ["p0", "升级", "响应流程", "响应"]):
            sections.append("\n## 升级提醒\n")
            sections.append("- 先确认影响范围、持续时间和是否命中核心链路。")
            sections.append("- 若已达到 P0/P1 级别，立即按值班链路升级，并同步相关负责人。")

        return "\n".join(sections).strip()

    def _offline_chat(self, message: str, history: List[ChatHistoryItem]) -> dict:
        documents, tool_logs = self._read_documents_for_offline_answer(message, history)
        return {
            "reply": self._build_offline_reply(message, documents),
            "tool_calls": tool_logs,
        }

    def _offline_stream_chat(self, message: str, history: List[ChatHistoryItem]) -> Iterator[str]:
        try:
            documents, tool_logs = self._read_documents_for_offline_answer(message, history)
            for event in tool_logs:
                yield encode_sse_event(event)
            yield encode_sse_event({"type": "status", "message": "助手正在整理 SOP 结论..."})
            reply = self._build_offline_reply(message, documents)
            yield encode_sse_event({"type": "reply_delta", "delta": reply})
            yield encode_sse_event({"type": "done", "reply": reply})
        except (GeneratorExit, asyncio.CancelledError):
            return

    def chat(self, message: str, history: List[ChatHistoryItem]) -> dict:
        if not self._has_api_key():
            return self._offline_chat(message, history)
        tools = self._tool_spec()
        candidates = self._rag_candidates(message, history)
        messages = self._build_lc_messages(history, message, candidates)
        tool_calls_log: List[dict] = []

        with self._lock:
            try:
                for _ in range(4):
                    response_payload = self._client.complete(self._serialize_lc_messages(messages), tools)
                    choices = response_payload.get("choices") or []
                    if not choices:
                        raise RuntimeError("DashScope chat response did not include any choices.")
                    choice = choices[0]
                    assistant_message = choice.get("message", {})
                    tool_calls = assistant_message.get("tool_calls") or []
                    content = assistant_message.get("content") or ""

                    if tool_calls:
                        self._append_assistant_tool_calls(messages, content, tool_calls)
                        for tool_call in tool_calls:
                            execution = self._execute_tool_call(tool_call)
                            tool_calls_log.extend(execution.events)
                            messages.append(execution.message)
                        continue

                    return {
                        "reply": content.strip() or "我暂时还没有整理出可靠结论。",
                        "tool_calls": tool_calls_log,
                    }
            except RuntimeError:
                return self._offline_chat(message, history)

        raise RuntimeError("The agent did not finish within the tool-call limit.")

    def stream_chat(self, message: str, history: List[ChatHistoryItem]) -> Iterator[str]:
        try:
            if not self._has_api_key():
                yield from self._offline_stream_chat(message, history)
                return
            tools = self._tool_spec()
            candidates = self._rag_candidates(message, history)
            messages = self._build_lc_messages(history, message, candidates)

            with self._lock:
                yield encode_sse_event({"type": "status", "message": "助手正在思考..."})
                for _ in range(4):
                    content_parts: List[str] = []
                    pending_tool_calls: Dict[int, dict] = {}

                    for chunk in self._client.stream(self._serialize_lc_messages(messages), tools):
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        delta = choice.get("delta", {})
                        delta_content = delta.get("content")
                        if delta_content:
                            content_parts.append(delta_content)
                            yield encode_sse_event({"type": "reply_delta", "delta": delta_content})

                        for tool_call in delta.get("tool_calls") or []:
                            index = int(tool_call.get("index", 0))
                            entry = pending_tool_calls.setdefault(
                                index,
                                {
                                    "id": tool_call.get("id", ""),
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                },
                            )
                            if tool_call.get("id"):
                                entry["id"] = tool_call["id"]
                            function_payload = tool_call.get("function", {})
                            if function_payload.get("name"):
                                entry["function"]["name"] = function_payload["name"]
                            if function_payload.get("arguments"):
                                entry["function"]["arguments"] += function_payload["arguments"]

                    if pending_tool_calls:
                        tool_calls = [pending_tool_calls[index] for index in sorted(pending_tool_calls)]
                        self._append_assistant_tool_calls(messages, "".join(content_parts), tool_calls)
                        for index in sorted(pending_tool_calls):
                            execution = self._execute_tool_call(pending_tool_calls[index])
                            for event in execution.events:
                                yield encode_sse_event(event)
                            messages.append(execution.message)
                        continue

                    reply = "".join(content_parts).strip()
                    yield encode_sse_event({"type": "done", "reply": reply})
                    return

            yield encode_sse_event({"type": "error", "message": "助手在工具调用轮次限制内未完成回答。"})
        except (GeneratorExit, asyncio.CancelledError):
            return
        except RuntimeError as exc:
            if self._has_api_key():
                try:
                    yield from self._offline_stream_chat(message, history)
                except (GeneratorExit, asyncio.CancelledError):
                    return
            else:
                yield encode_sse_event({"type": "error", "message": str(exc)})
