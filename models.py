# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Data models for the Blood Pathology Environment."""

from typing import Dict, Any
from openenv.core.env_server.types import Action, Observation
from pydantic import Field

class PathologyAction(Action):
    """Action for the Pathology environment - executing a simulated LIMS tool."""

    command: str = Field(
        ..., 
        description=(
            "Tool name. Valid: 'list_pending_cases', 'get_patient_demographics', "
            "'get_medications', 'get_lab_orders', 'get_lab_results', "
            "'get_previous_results', 'query_reference_ranges', "
            "'flag_critical_value', 'submit_diagnostic_report'"
        )
    )
    arguments: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Arguments dict. E.g. {'patient_id': 1001} or {'order_id': 'ORD-E001'}"
    )

class PathologyObservation(Observation):
    """Observation from the Pathology environment - LIMS API output."""

    output: str = Field(default="", description="Structured JSON or text response from the LIMS.")
    error: str = Field(default="", description="Error message if the command failed, else empty.")
