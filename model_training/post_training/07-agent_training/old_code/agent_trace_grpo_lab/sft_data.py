"""Prepare and validate conversational agent traces for SFT."""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# These patterns are intentionally conservative; review outputs before use.
_REDACTIONS = [
    re.compile(r'(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+'),
    re.compile(r'(?i)((?:api[_-]?key|access[_-]?token|password|secret)\s*[=:]\s*["\']?)[^\s"\',;]+'),
    re.compile(r'\b(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b'),
    re.compile(r'-----BEGIN [^-]*PRIVATE KEY-----[\s\S]*?-----END [^-]*PRIVATE KEY-----'),
    re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
]
_REASONING = {'reasoning', 'reasoning_content', 'thinking', 'analysis', 'chain_of_thought'}


def redact(value: Any) -> Any:
    """Redact common sensitive strings recursively and omit reasoning fields."""
    if isinstance(value, str):
        for pattern in _REDACTIONS:
            value = pattern.sub(lambda m: (m.group(1) if m.lastindex else '') + '[REDACTED]', value)
        return value
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            key_name = str(key).lower()
            if key_name in _REASONING:
                continue
            if key_name in {'password', 'secret', 'api_key', 'access_token', 'authorization'} and isinstance(item, str) and item:
                result[key] = '[REDACTED]'
            else:
                result[key] = redact(item)
        return result
    return value


def normalize_trace(trace: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Convert a shared AgentTrace to a TRL record and indexed validation issues."""
    logger.debug('Normalizing trace with %d messages', len(trace.messages))
    messages, issues, pending, seen = [], [], {}, set()
    for message in trace.messages:
        index = message.index
        role = message.role
        out = {'role': role, 'content': redact(message.content)}
        if role not in {'system', 'user', 'assistant', 'tool'}:
            issues.append({'type': 'invalid_role', 'message_index': index})
        if message.name:
            out['name'] = redact(message.name)
        if message.tool_calls:
            if role != 'assistant':
                issues.append({'type': 'call_on_non_assistant', 'message_index': index})
            calls = []
            for call in message.tool_calls:
                args = redact(call.arguments)
                if not call.id or not call.name or call.arguments_error or not isinstance(args, dict):
                    issues.append({'type': 'malformed_tool_call', 'message_index': index})
                if call.id:
                    if call.id in seen:
                        issues.append({'type': 'duplicate_tool_call_id', 'message_index': index})
                    seen.add(call.id)
                    pending[call.id] = index
                calls.append({'id': call.id, 'type': 'function', 'function': {'name': call.name, 'arguments': args}})
            out['tool_calls'] = calls
        if role == 'tool':
            out['tool_call_id'] = message.tool_call_id
            if message.tool_call_id in pending:
                del pending[message.tool_call_id]
            else:
                issues.append({'type': 'unmatched_tool_response', 'message_index': index})
        messages.append(out)
    for index in pending.values():
        issues.append({'type': 'unmatched_tool_call', 'message_index': index})
    if not messages or messages[-1]['role'] != 'assistant' or messages[-1].get('tool_calls'):
        issues.append({'type': 'incomplete_trajectory', 'message_index': len(messages) - 1})
    result = {'messages': messages}
    if trace.metadata.get('tools') is not None:
        result['tools'] = redact(trace.metadata['tools'])
    logger.debug('Normalized trace: messages=%d issues=%d', len(messages), len(issues))
    return result, issues


def write_sft_dataset(records: Iterable[dict[str, Any]], adapter: Any, output_path: Path, summary_path: Path, issues_path: Path) -> dict[str, Any]:
    """Write all normalized records, separate validation issues, and aggregate counts."""
    logger.info('Preparing SFT records at %s', output_path)
    paths = [Path(output_path), Path(summary_path), Path(issues_path)]
    if len(set(paths)) != len(paths) or any(path.exists() for path in paths):
        raise FileExistsError('Output paths must be distinct and not already exist')
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    roles = Counter()
    issue_counts = Counter()
    with paths[0].open('w', encoding='utf-8') as output, paths[2].open('w', encoding='utf-8') as details:
        for row_index, row in enumerate(records):
            normalized, issues = normalize_trace(adapter.parse(row))
            output.write(json.dumps(normalized, ensure_ascii=False) + '\n')
            details.write(json.dumps({'row_index': row_index, 'issues': issues}) + '\n')
            counts['total_trajectories'] += 1
            counts['invalid_records' if issues else 'valid_records'] += 1
            roles.update(str(message['role']) for message in normalized['messages'])
            issue_counts.update(issue['type'] for issue in issues)
    summary = {key: counts[key] for key in ('total_trajectories', 'valid_records', 'invalid_records')}
    summary.update(unmatched_tool_calls=issue_counts['unmatched_tool_call'], message_role_distribution=dict(roles), issue_counts=dict(issue_counts))
    paths[1].write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    logger.info('Prepared %d records: %d invalid', counts['total_trajectories'], counts['invalid_records'])
    return summary
