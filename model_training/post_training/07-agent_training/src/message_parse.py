from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def build_tool_interactions(
    sample: Mapping[str, Any] | pd.Series,
    messages: Sequence[Mapping[str, Any]] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Return all trace messages, pairing assistant tool calls with responses.

    System, user, and other non-interaction messages appear once with their
    original fields. An assistant message with multiple non-``finish`` calls
    appears once per call, retaining the same ``message_index``. Tool replies
    are matched by call ID; unmatched tool messages remain standalone rows.
    The input message objects are not modified.
    """
    if messages is None:
        messages = sample.get("messages")
    if messages is None:
        return pd.DataFrame(
            columns=["message_index", "row_type", "message", "tool_call"]
        )

    message_list = (
        messages.tolist() if isinstance(messages, np.ndarray) else list(messages)
    )
    response_by_id: dict[Any, list[tuple[int, Mapping[str, Any]]]] = {}
    calls_by_message: dict[int, list[Mapping[str, Any]]] = {}
    call_ids: set[Any] = set()

    for message_index, message in enumerate(message_list):
        if not isinstance(message, Mapping):
            continue
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if call_id is not None:
                response_by_id.setdefault(call_id, []).append((message_index, message))
        if message.get("role") != "assistant":
            continue

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, np.ndarray):
            tool_calls = tool_calls.tolist()
        elif isinstance(tool_calls, Mapping):
            tool_calls = [tool_calls]
        if not isinstance(tool_calls, (list, tuple)):
            continue

        relevant_calls = [
            call
            for call in tool_calls
            if isinstance(call, Mapping)
            and not (
                isinstance(call.get("function"), Mapping)
                and call["function"].get("name") == "finish"
            )
        ]
        if relevant_calls:
            calls_by_message[message_index] = relevant_calls
            call_ids.update(
                call.get("id")
                for call in relevant_calls
                if call.get("id") is not None
            )

    rows: list[dict[str, Any]] = []
    for message_index, message in enumerate(message_list):
        if not isinstance(message, Mapping):
            rows.append({
                "message_index": message_index,
                "row_type": "message",
                "message": message,
            })
            continue

        calls = calls_by_message.get(message_index)
        if calls:
            for call in calls:
                function = call.get("function")
                if not isinstance(function, Mapping):
                    function = {}
                call_id = call.get("id")
                matches = response_by_id.get(call_id, []) if call_id is not None else []
                row = {
                    **message,
                    "message_index": message_index,
                    "row_type": "tool_interaction",
                    "message": message,
                    "tool_call": call,
                    "call_id": call_id,
                    "call_type": call.get("type"),
                    "call_name": function.get("name"),
                    "call_arguments": function.get("arguments"),
                    "response_message_index": (
                        matches[0][0] if len(matches) == 1 else None
                    ),
                    "response_message": (
                        matches[0][1] if len(matches) == 1 else None
                    ),
                    "response_matches": [
                        {"message_index": index, "message": response}
                        for index, response in matches
                    ],
                    "matched": len(matches) == 1,
                }
                if len(matches) == 1:
                    row.update(
                        {
                            f"response_{key}": value
                            for key, value in matches[0][1].items()
                        }
                    )
                rows.append(row)
            continue

        message_row = {
            **message,
            "message_index": message_index,
            "row_type": "message",
            "message": message,
        }
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if call_id in call_ids:
                continue
        rows.append(message_row)

    return pd.DataFrame(rows)
