"""Adapter for SWE-Gym OpenHands sampled trajectories."""

import json
import logging
import re
from typing import Any, Mapping

from trace_schema import AgentTrace, ToolCall, TraceMessage

logger = logging.getLogger(__name__)


def parse_run_id(run_id: str) -> dict[str, Any]:
    """Extract known run ID fields, preserving unrecognized IDs."""
    fields = {"run_id": run_id, "model_name": None, "model_date": None,
              "max_iterations": None, "run_config": None, "run_number": None}
    match = re.fullmatch(r"(.+?)_maxiter_(\d+)_(.+)-run_(\d+)", run_id)
    if match:
        model_id, max_iterations, run_config, run_number = match.groups()
        model = re.fullmatch(r"(.+)-(\d{4}-\d{2}-\d{2}|\d{8})", model_id)
        fields.update(model_name=model.group(1) if model else model_id,
                      model_date=model.group(2) if model else None,
                      max_iterations=int(max_iterations), run_config=run_config,
                      run_number=int(run_number))
    return fields


def parse_instance_id(instance_id: str) -> dict[str, Any]:
    """Extract repository and issue number, retaining the original ID."""
    fields = {"instance_id": instance_id, "owner": None, "repository": None,
              "issue_number": None}
    match = re.fullmatch(r"(.+?)__(.+)-(\d+)", instance_id)
    if match:
        owner, repository, issue_number = match.groups()
        fields.update(owner=owner, repository=repository, issue_number=int(issue_number))
    return fields


def _parse_call(call: Mapping[str, Any]) -> ToolCall:
    if not isinstance(call, Mapping):
        return ToolCall(None, None, call, "Tool call must be a mapping")
    function = call.get("function") or {}
    if not isinstance(function, Mapping):
        return ToolCall(call.get("id"), None, function, "Function must be a mapping")
    arguments = function.get("arguments")
    error = None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            error = str(exc)
            logger.debug("Invalid tool arguments: %s", exc)
    return ToolCall(call.get("id"), function.get("name"), arguments, error)


class SWEGymTraceAdapter:
    """Convert OpenHands messages and evaluation fields into an agent trace."""

    def parse(self, record: Mapping[str, Any]) -> AgentTrace:
        """Parse one SWE-Gym record without adding labels to its messages."""
        messages = [
            TraceMessage(index=index, role=message.get("role"),
                         content=message.get("content"),
                         tool_call_id=message.get("tool_call_id"),
                         tool_calls=[_parse_call(call) for call in (message.get("tool_calls") or [])],
                         name=message.get("name"))
            for index, message in enumerate(record.get("messages") or [])
        ]
        metadata = {"run": parse_run_id(record.get("run_id") or ""),
                    "instance": parse_instance_id(record.get("instance_id") or ""),
                    "tools": record.get("tools")}
        labels = {"resolved": record.get("resolved"), "test_result": record.get("test_result")}
        logger.debug("Parsed SWE-Gym trace: messages=%d", len(messages))
        return AgentTrace(messages=messages, metadata=metadata, labels=labels)
