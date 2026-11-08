"""Public trace inspection types and dataset adapters."""

from swe_gym_trace import SWEGymTraceAdapter, parse_instance_id, parse_run_id
from trace_analysis import (join_trace_tasks, plot_message_distributions, plot_trace_distributions,
                            structure_trace_ids, summarize_trace_messages)
from trace_schema import AgentTrace, ToolCall, TraceAdapter, TraceMessage
from sft_data import normalize_trace, redact, write_sft_dataset

__all__ = ["AgentTrace", "ToolCall", "TraceAdapter", "TraceMessage",
           "SWEGymTraceAdapter", "parse_instance_id", "parse_run_id",
           "join_trace_tasks", "plot_message_distributions", "plot_trace_distributions",
           "structure_trace_ids", "summarize_trace_messages", "normalize_trace", "redact", "write_sft_dataset"]
