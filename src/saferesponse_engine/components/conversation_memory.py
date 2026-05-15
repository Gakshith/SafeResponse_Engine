import json
import re
import time
import uuid
import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from saferesponse_engine.entity.config_entity import ConversationMemoryConfig


class ConversationMemoryLayer:
    def __init__(self, config: ConversationMemoryConfig):
        self.config = config

    @contextmanager
    def _file_lock(self, exclusive: bool):
        path = Path(self.config.memory_store_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        with lock_path.open("w", encoding="utf-8") as lock_file:
            operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(lock_file.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _load_store_unlocked(self) -> dict[str, Any]:
        path = Path(self.config.memory_store_path)
        if not path.exists():
            return {"conversations": {}}
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_store(self) -> dict[str, Any]:
        with self._file_lock(exclusive=False):
            return self._load_store_unlocked()

    def _save_store_unlocked(self, store: dict[str, Any]) -> None:
        path = Path(self.config.memory_store_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(store, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temp_path.replace(path)

    def _save_store(self, store: dict[str, Any]) -> None:
        with self._file_lock(exclusive=True):
            self._save_store_unlocked(store)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "about", "briefly", "give", "how", "i", "in", "is", "it", "me",
            "of", "on", "or", "short", "summary", "tell",
            "that", "the", "this", "to", "us", "was", "we", "what", "who",
            "why", "with", "you",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in stopwords and len(token) > 1
        }

    @staticmethod
    def _trim_words(text: str, max_words: int) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text.strip()
        return " ".join(words[:max_words]).rstrip() + "..."

    def get_history(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._file_lock(exclusive=False):
            store = self._load_store_unlocked()
        conversation = store["conversations"].get(conversation_id, {})
        return conversation.get("turns", [])

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._file_lock(exclusive=False):
            store = self._load_store_unlocked()
        conversation = store["conversations"].get(conversation_id)
        if conversation is None:
            return None
        return {
            "conversation_id": conversation_id,
            "turns": conversation.get("turns", []),
        }

    @staticmethod
    def _conversation_title(turns: list[dict[str, Any]]) -> str:
        for turn in turns:
            if turn.get("role") == "user" and turn.get("text", "").strip():
                return ConversationMemoryLayer._trim_words(turn["text"], 8)
        return "New chat"

    def list_conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._file_lock(exclusive=False):
            store = self._load_store_unlocked()
        conversations = []
        for conversation_id, conversation in store.get("conversations", {}).items():
            turns = conversation.get("turns", [])
            if not turns:
                continue
            last_turn = turns[-1]
            user_turns = [turn for turn in turns if turn.get("role") == "user"]
            conversations.append(
                {
                    "conversation_id": conversation_id,
                    "title": self._conversation_title(turns),
                    "preview": self._trim_words(last_turn.get("text", ""), 16),
                    "updated_at": last_turn.get("timestamp"),
                    "turn_count": len(turns),
                    "message_count": len(user_turns),
                }
            )

        conversations.sort(
            key=lambda item: item.get("updated_at") or 0,
            reverse=True,
        )
        return conversations[:limit]

    def _prune_store_unlocked(self, store: dict[str, Any]) -> None:
        max_conversations = max(1, int(self.config.max_conversations))
        conversations = store.get("conversations", {})
        if len(conversations) <= max_conversations:
            return

        ordered = sorted(
            conversations.items(),
            key=lambda item: (
                item[1].get("turns", [{}])[-1].get("timestamp", 0)
                if item[1].get("turns")
                else 0
            ),
        )
        for conversation_id, _conversation in ordered[: len(conversations) - max_conversations]:
            conversations.pop(conversation_id, None)

    def create_conversation_id(self) -> str:
        return str(uuid.uuid4())

    def append_turn(
        self,
        conversation_id: str,
        role: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._file_lock(exclusive=True):
            store = self._load_store_unlocked()
            conversation = store["conversations"].setdefault(
                conversation_id,
                {"conversation_id": conversation_id, "turns": []},
            )
            turn = {
                "turn_id": len(conversation["turns"]) + 1,
                "role": role,
                "text": text.strip(),
                "timestamp": time.time(),
                "metadata": metadata or {},
            }
            conversation["turns"].append(turn)
            self._prune_store_unlocked(store)
            self._save_store_unlocked(store)
        return turn

    def _build_summary(self, history: list[dict[str, Any]]) -> str:
        recent_turns = history[-self.config.max_recent_turns:]
        lines = [
            f"Turn {turn['turn_id']} {turn['role']}: {turn.get('text', '')}"
            for turn in recent_turns
        ]
        return self._trim_words(" ".join(lines), self.config.summary_max_words)

    def _select_relevant_turns(
        self,
        history: list[dict[str, Any]],
        query: str,
    ) -> list[dict[str, Any]]:
        query_tokens = self._tokens(query)
        if not query_tokens:
            return []

        scored_turns = []
        for turn in history:
            if turn.get("role") != "assistant":
                continue
            decision = turn.get("metadata", {}).get("decision")
            if decision not in {"ACCEPT", "DIRECT"}:
                continue
            turn_tokens = self._tokens(turn.get("text", ""))
            if not turn_tokens:
                continue
            overlap = len(query_tokens & turn_tokens) / len(query_tokens)
            if overlap >= self.config.min_overlap_score:
                scored_turns.append((overlap, turn))

        scored_turns.sort(key=lambda item: (-item[0], item[1]["turn_id"]))
        return [
            turn
            for _, turn in scored_turns[: self.config.max_relevant_turns]
        ]

    def build_context(
        self,
        conversation_id: str,
        query: str,
    ) -> dict[str, Any]:
        history = self.get_history(conversation_id)
        user_turns = [turn for turn in history if turn.get("role") == "user"]
        mode = "single_turn" if not user_turns else "memory_augmented"

        if mode == "single_turn":
            return {
                "conversation_id": conversation_id,
                "mode": mode,
                "memory_context": "",
                "summary": "",
                "relevant_turns": [],
            }

        relevant_turns = self._select_relevant_turns(history, query)
        summary = self._build_summary(history)
        if relevant_turns:
            relevant_lines = [
                f"Turn {turn['turn_id']} {turn['role']}: {turn.get('text', '')}"
                for turn in relevant_turns
            ]
            memory_context = (
                "Conversation summary:\n"
                f"{summary}\n\n"
                "Relevant original turns:\n"
                + "\n".join(relevant_lines)
            )
        else:
            memory_context = ""

        return {
            "conversation_id": conversation_id,
            "mode": mode,
            "memory_context": memory_context,
            "summary": summary,
            "relevant_turns": relevant_turns,
        }
