from pathlib import Path

from saferesponse_engine.components.conversation_memory import (
    ConversationMemoryLayer,
)
from saferesponse_engine.entity.config_entity import ConversationMemoryConfig


def _config(tmp_path: Path) -> ConversationMemoryConfig:
    return ConversationMemoryConfig(
        root_dir=tmp_path,
        memory_store_path=tmp_path / "conversations.json",
        max_recent_turns=6,
        max_relevant_turns=4,
        summary_max_words=120,
        min_overlap_score=0.05,
        max_conversations=200,
    )


def test_memory_context_is_blank_without_relevant_turns(tmp_path):
    memory = ConversationMemoryLayer(_config(tmp_path))
    conversation_id = "demo"
    memory.append_turn(conversation_id, "user", "Tell me about Mercury.")
    memory.append_turn(
        conversation_id,
        "assistant",
        "I cannot provide a reliable answer.",
    )

    context = memory.build_context(conversation_id, "Tell me briefly who Aristotle was.")

    assert context["mode"] == "memory_augmented"
    assert context["relevant_turns"] == []
    assert context["memory_context"] == ""


def test_memory_context_includes_relevant_turns(tmp_path):
    memory = ConversationMemoryLayer(_config(tmp_path))
    conversation_id = "demo"
    memory.append_turn(
        conversation_id,
        "user",
        "Give me a short summary about Abraham Lincoln.",
    )
    memory.append_turn(
        conversation_id,
        "assistant",
        "Abraham Lincoln was the 16th president.",
        metadata={"decision": "ACCEPT"},
    )

    context = memory.build_context(conversation_id, "What else about Lincoln?")

    assert context["mode"] == "memory_augmented"
    assert context["relevant_turns"]
    assert "Abraham Lincoln" in context["memory_context"]


def test_conversation_listing_and_loading(tmp_path):
    memory = ConversationMemoryLayer(_config(tmp_path))
    first_id = "first"
    second_id = "second"

    memory.append_turn(first_id, "user", "Tell me about Mercury and its orbit.")
    memory.append_turn(first_id, "assistant", "Mercury is the closest planet.")
    memory.append_turn(second_id, "user", "Who was Abraham Lincoln?")

    conversations = memory.list_conversations()
    loaded = memory.get_conversation(first_id)

    assert conversations[0]["conversation_id"] == second_id
    assert conversations[1]["title"] == "Tell me about Mercury and its orbit."
    assert conversations[1]["message_count"] == 1
    assert loaded["turns"][0]["text"] == "Tell me about Mercury and its orbit."
    assert memory.get_conversation("missing") is None


def test_conversation_store_prunes_oldest_conversations(tmp_path):
    config = ConversationMemoryConfig(
        root_dir=tmp_path,
        memory_store_path=tmp_path / "conversations.json",
        max_recent_turns=6,
        max_relevant_turns=4,
        summary_max_words=120,
        min_overlap_score=0.05,
        max_conversations=1,
    )
    memory = ConversationMemoryLayer(config)

    memory.append_turn("first", "user", "First question")
    memory.append_turn("second", "user", "Second question")

    conversations = memory.list_conversations()

    assert len(conversations) == 1
    assert conversations[0]["conversation_id"] == "second"
    assert memory.get_conversation("first") is None
