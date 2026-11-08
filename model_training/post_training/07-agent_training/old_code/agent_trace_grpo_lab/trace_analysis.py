"""Data preparation and plots for exploring agent trajectories."""

import logging
from collections import Counter
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from swe_gym_trace import parse_instance_id, parse_run_id

logger = logging.getLogger(__name__)


def structure_trace_ids(traces: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attach parsed IDs and return attempts, distinct runs, and distinct issues."""
    run_df = pd.DataFrame([parse_run_id(value) for value in traces["run_id"].dropna().unique()],
                          columns=["run_id", "model_name", "model_date", "max_iterations", "run_config", "run_number"])
    instance_df = pd.DataFrame([parse_instance_id(value) for value in traces["instance_id"].dropna().unique()],
                               columns=["instance_id", "owner", "repository", "issue_number"])
    result = traces.merge(run_df, on="run_id", how="left").merge(instance_df, on="instance_id", how="left")
    logger.info("Structured %d traces into %d runs and %d issues", len(result), len(run_df), len(instance_df))
    return result, run_df, instance_df


def join_trace_tasks(traces: pd.DataFrame, tasks: pd.DataFrame) -> pd.DataFrame:
    """Left-join each trace to one task while retaining original trace positions."""
    trace_rows = traces.reset_index(drop=True).rename_axis("trace_row").reset_index()
    task_rows = tasks[tasks["instance_id"].notna()].rename(
        columns={column: f"task_{column}" for column in tasks.columns if column != "instance_id"}
    )
    result = trace_rows.merge(task_rows, on="instance_id", how="left", validate="many_to_one",
                              indicator="task_match", sort=False)
    logger.info("Joined %d traces: matched=%d unmatched=%d", len(result),
                int((result["task_match"] == "both").sum()), int((result["task_match"] == "left_only").sum()))
    return result


def summarize_trace_messages(records: Iterable[Mapping[str, Any]]) -> tuple[list[int], Counter, Counter]:
    """Count messages per trace, roles, and actual tool calls in source records."""
    lengths: list[int] = []
    roles: Counter = Counter()
    tools: Counter = Counter()
    for record in records:
        messages = record["messages"] or []
        lengths.append(len(messages))
        for message in messages:
            roles[message.get("role") or "Unknown"] += 1
            for call in message.get("tool_calls") or []:
                tools[(call.get("function") or {}).get("name") or "Unknown"] += 1
    logger.info("Summarized %d traces: messages=%d tool_calls=%d", len(lengths), sum(lengths), sum(tools.values()))
    return lengths, roles, tools


def plot_trace_distributions(traces: pd.DataFrame) -> Any:
    """Plot outcomes, models, top repositories and iteration limits."""
    data = traces.copy()
    data["outcome"] = data["resolved"].map({True: "Resolved", False: "Not resolved"}).fillna("Unknown")
    data["model_name"] = data["model_name"].fillna("Unknown")
    data["repo"] = data["owner"].fillna("Unknown") + "/" + data["repository"].fillna("Unknown")
    repo_counts = data["repo"].value_counts().head(15)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.countplot(data=data, x="outcome", ax=axes[0, 0], color="steelblue")
    axes[0, 0].set(title="Evaluation outcomes", xlabel="Outcome", ylabel="Attempts")
    sns.countplot(data=data, y="model_name", hue="outcome", ax=axes[0, 1])
    axes[0, 1].set(title="Outcomes by model", xlabel="Attempts", ylabel="Model")
    sns.barplot(x=repo_counts.values, y=repo_counts.index, ax=axes[1, 0], color="steelblue")
    axes[1, 0].set(title="Top 15 repositories", xlabel="Attempts", ylabel="Repository")
    sns.countplot(data=data, x="max_iterations", ax=axes[1, 1], color="steelblue")
    axes[1, 1].set(title="Iteration limits", xlabel="Maximum iterations", ylabel="Attempts")
    fig.tight_layout()
    return fig


def plot_message_distributions(lengths: list[int], roles: Counter, tools: Counter) -> Any:
    """Plot trace lengths, message roles, and the most frequent tool calls."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    sns.histplot(lengths, bins=30, ax=axes[0], color="steelblue")
    axes[0].set(title="Trace lengths", xlabel="Messages per attempt", ylabel="Attempts")
    role_counts = pd.Series(roles, dtype="int64").sort_values(ascending=False)
    sns.barplot(x=role_counts.values, y=role_counts.index, ax=axes[1], color="steelblue")
    axes[1].set(title="Message roles", xlabel="Messages", ylabel="Role")
    tool_counts = pd.Series(tools, dtype="int64").sort_values(ascending=False).head(15)
    if not tool_counts.empty:
        sns.barplot(x=tool_counts.values, y=tool_counts.index, ax=axes[2], color="steelblue")
    else:
        axes[2].text(0.5, 0.5, "No tool calls recorded", ha="center", va="center", transform=axes[2].transAxes)
    axes[2].set(title="Top 15 tools called", xlabel="Tool calls", ylabel="Tool")
    fig.tight_layout()
    return fig
