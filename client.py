# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Pathology Env Client."""

from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import PathologyAction, PathologyObservation

class PathologyEnv(
    EnvClient[PathologyAction, PathologyObservation, State]
):
    """Client for the Pathology Environment."""

    def _step_payload(self, action: PathologyAction) -> Dict:
        return {
            "command": action.command,
            "arguments": action.arguments,
        }

    def _parse_result(self, payload: Dict) -> StepResult[PathologyObservation]:
        obs_data = payload.get("observation", {})
        observation = PathologyObservation(
            output=obs_data.get("output", ""),
            error=obs_data.get("error", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
