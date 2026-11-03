from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def prepare_message_df(
    sample: Mapping[str, Any] | pd.Series,
    messages: Sequence[Mapping[str, Any]] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Create one row per source message while retaining its original fields."""
    if messages is None:
        messages = sample.get("messages")
    if messages is None:
        return pd.DataFrame(
            columns=["instance_id", "run_id", "message_index", "row_type", "message"]
        )

    message_list = (
        messages.tolist() if isinstance(messages, np.ndarray) else list(messages)
    )
    rows = []
    for message_index, message in enumerate(message_list):
        row = dict(message) if isinstance(message, Mapping) else {}
        row.update({
            "instance_id": sample.get("instance_id"),
            "run_id": sample.get("run_id"),
            "message_index": message_index,
            "row_type": "message",
            "message": message,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def extract_tool_call_df(message_df: pd.DataFrame) -> pd.DataFrame:
    """Extract non-finish assistant calls into one row per call."""
    rows = []
    for _, message_row in message_df.iterrows():
        message = message_row.get("message")
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, np.ndarray):
            tool_calls = tool_calls.tolist()
        elif isinstance(tool_calls, Mapping):
            tool_calls = [tool_calls]
        if not isinstance(tool_calls, (list, tuple)):
            continue

        for call in tool_calls:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            if not isinstance(function, Mapping):
                function = {}
            if function.get("name") == "finish":
                continue

            row = dict(message_row)
            row.update({
                "row_type": "tool_interaction",
                "tool_call": call,
                "call_id": call.get("id"),
                "call_type": call.get("type"),
                "call_name": function.get("name"),
                "call_arguments": function.get("arguments"),
            })
            rows.append(row)

    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=[
        *message_df.columns,
        "tool_call",
        "call_id",
        "call_type",
        "call_name",
        "call_arguments",
    ])


def attach_tool_responses(
    message_df: pd.DataFrame,
    tool_call_df: pd.DataFrame,
) -> pd.DataFrame:
    """Match calls to tool replies and combine them with remaining messages."""
    response_by_id: dict[Any, list[tuple[int, Mapping[str, Any]]]] = {}
    for _, row in message_df.iterrows():
        message = row.get("message")
        if not isinstance(message, Mapping) or message.get("role") != "tool":
            continue
        call_id = message.get("tool_call_id")
        if call_id is not None:
            response_by_id.setdefault(call_id, []).append(
                (int(row["message_index"]), message)
            )

    interaction_rows = []
    matched_response_indices = set()
    for row in tool_call_df.to_dict("records"):
        call_id = row.get("call_id")
        matches = response_by_id.get(call_id, []) if call_id is not None else []
        matched_response_indices.update(index for index, _ in matches)
        row.update({
            "response_message_index": matches[0][0] if len(matches) == 1 else None,
            "response_message": matches[0][1] if len(matches) == 1 else None,
            "response_matches": [
                {"message_index": index, "message": message}
                for index, message in matches
            ],
            "matched": len(matches) == 1,
        })
        if len(matches) == 1:
            row.update({
                f"response_{key}": value
                for key, value in matches[0][1].items()
            })
        interaction_rows.append(row)

    if not interaction_rows:
        return message_df.reset_index(drop=True)

    call_message_indices = set(tool_call_df["message_index"].tolist())
    keep_rows = []
    for row in message_df.to_dict("records"):
        if row.get("message_index") in call_message_indices:
            continue
        if row.get("message_index") in matched_response_indices:
            continue
        keep_rows.append(row)

    result = pd.concat(
        [pd.DataFrame(keep_rows), pd.DataFrame(interaction_rows)],
        ignore_index=True,
        sort=False,
    )
    return result.sort_values("message_index", kind="stable").reset_index(drop=True)


def build_tool_interactions(
    sample: Mapping[str, Any] | pd.Series,
    messages: Sequence[Mapping[str, Any]] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Run the three message parsing steps and return the combined DataFrame."""
    message_df = prepare_message_df(sample, messages)
    tool_call_df = extract_tool_call_df(message_df)
    return attach_tool_responses(message_df, tool_call_df)
