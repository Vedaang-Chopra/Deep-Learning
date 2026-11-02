"""SWE-Gym OpenHands identifier parsing."""

import logging
import re
from typing import Any

import pandas as pd

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


def structure_trace_ids(traces: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attach parsed IDs and return attempts, distinct runs, and distinct issues."""
    run_df = pd.DataFrame([parse_run_id(value) for value in traces["run_id"].dropna().unique()],
                          columns=["run_id", "model_name", "model_date", "max_iterations", "run_config", "run_number"])
    instance_df = pd.DataFrame([parse_instance_id(value) for value in traces["instance_id"].dropna().unique()],
                               columns=["instance_id", "owner", "repository", "issue_number"])
    result = traces.merge(run_df, on="run_id", how="left").merge(instance_df, on="instance_id", how="left")
    logger.info("Structured %d traces into %d runs and %d issues", len(result), len(run_df), len(instance_df))
    return result, run_df, instance_df
