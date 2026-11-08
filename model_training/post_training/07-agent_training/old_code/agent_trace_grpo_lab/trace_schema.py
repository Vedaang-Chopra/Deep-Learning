"""Dataset-independent types for inspecting agent traces."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass
class ToolCall:
    """A tool call with its original ID and decoded or raw arguments."""

    id: str | None
    name: str | None
    arguments: Any
    arguments_error: str | None = None


@dataclass
class TraceMessage:
    """One message in the source order, including any tool calls."""

    index: int
    role: str | None
    content: Any
    tool_call_id: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    name: str | None = None


@dataclass
class AgentTrace:
    """Messages, source-specific metadata, and separate evaluation labels."""

    messages: list[TraceMessage]
    metadata: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)


class TraceAdapter(Protocol):
    """Convert a source dataset record into the shared trace representation."""

    def parse(self, record: Mapping[str, Any]) -> AgentTrace:
        """Convert a single record into an agent trace."""
        ...
