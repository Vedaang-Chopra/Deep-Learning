from collections.abc import Mapping, Sequence
import json
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_finish_call(messages: Any) -> bool | None:
    """Detect a finish call in assistant messages; return None for incomplete input."""
    logger.debug("detect_finish_call | messages_type=%s", type(messages).__name__)
    if not isinstance(messages, (list, tuple, np.ndarray)):
        logger.debug("detect_finish_call | finish_called=unknown")
        return None

    def find_call(value: Any) -> bool:
        if isinstance(value, Mapping):
            function = value.get("function")
            if isinstance(function, Mapping) and function.get("name") == "finish":
                return True
            return any(
                find_call(item)
                for key, item in value.items()
                if key not in {"function", "arguments"}
            )
        if isinstance(value, (list, tuple, np.ndarray)):
            return any(find_call(item) for item in value)
        return False

    incomplete = False
    for message in messages:
        if not isinstance(message, Mapping):
            incomplete = True
            continue
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if calls is not None and not isinstance(calls, (Mapping, list, tuple, np.ndarray)):
            incomplete = True
        if find_call(calls):
            logger.debug("detect_finish_call | finish_called=True")
            return True
        legacy_call = message.get("function_call")
        if isinstance(legacy_call, Mapping) and legacy_call.get("name") == "finish":
            logger.debug("detect_finish_call | finish_called=True")
            return True
    result = None if incomplete else False
    logger.debug("detect_finish_call | finish_called=%s", result)
    return result


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


def enrich_tool_interactions(
    interaction_df: pd.DataFrame,
    tools: Sequence[Mapping[str, Any]] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Expand call arguments and attach tool definitions by function name."""
    logger.info("enrich_tool_interactions | input_rows=%d", len(interaction_df))
    result = interaction_df.reset_index(drop=True).copy()
    arguments = result.get("call_arguments", pd.Series([None] * len(result)))
    argument_dicts = []
    errors = []
    for value in arguments:
        error = None
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                error = str(exc)
                value = None
        if isinstance(value, Mapping):
            argument_dicts.append(dict(value))
        else:
            argument_dicts.append({})
            if error is None and value is not None and value is not pd.NA:
                if not (isinstance(value, (float, np.floating)) and np.isnan(value)):
                    error = "Tool call arguments must be a JSON object."
        errors.append(error)

    argument_df = pd.json_normalize(argument_dicts).add_prefix("arg_")
    result = result.join(argument_df)
    result["argument_parse_error"] = errors
    error_count = sum(error is not None for error in errors)
    if error_count:
        logger.warning("enrich_tool_interactions | argument_parse_errors=%d", error_count)

    tool_list = [] if tools is None else list(tools)
    definitions = pd.json_normalize(
        [tool for tool in tool_list if isinstance(tool, Mapping)]
    ).add_prefix("tool_definition_")
    name_column = "tool_definition_function.name"
    if name_column in definitions.columns and "call_name" in result.columns:
        definitions = definitions[definitions[name_column].notna()]
        result = result.merge(
            definitions,
            how="left",
            left_on="call_name",
            right_on=name_column,
            sort=False,
            validate="many_to_one",
        )
    logger.info("enrich_tool_interactions | output_rows=%d", len(result))
    return result


def build_tool_interactions(
    sample: Mapping[str, Any] | pd.Series,
    messages: Sequence[Mapping[str, Any]] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Run the message parsing steps and return the combined DataFrame."""
    message_df = prepare_message_df(sample, messages)
    tool_call_df = extract_tool_call_df(message_df)
    interaction_df = attach_tool_responses(message_df, tool_call_df)
    return enrich_tool_interactions(interaction_df, sample.get("tools"))
