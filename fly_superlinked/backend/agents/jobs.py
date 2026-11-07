"""The everyday agent jobs moved off a frontier model: mission planning and perception-to-facts.

Both use a strict JSON schema, validated with pydantic; invalid output is retried once, then the
job fails loudly (CallResult.ok is False and the error is surfaced in the UI and bench).
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..sim import arena
from . import llm

LANDMARK_NAMES = [lm.name for lm in arena.LANDMARKS]
PLACE_NAMES = LANDMARK_NAMES + ["home"]
DRONES = ["scout-1", "scout-2", "courier"]
ZONES = ["pond", "gust_zone"] + LANDMARK_NAMES


class Goal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    landmark: Literal[tuple(PLACE_NAMES)]
    action: Literal["visit", "survey", "deliver"]


class Constraint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["avoid"]
    zone: Literal[tuple(ZONES)]


class DronePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drone: Literal[tuple(DRONES)]
    goals: List[Goal]
    constraints: List[Constraint]
    return_home: bool


class MissionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drones: List[DronePlan] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_drones(self):
        names = [d.drone for d in self.drones]
        if len(names) != len(set(names)):
            raise ValueError("each drone may appear once")
        return self


def plan_schema():
    s = MissionPlan.model_json_schema()
    return _inline_refs(s)


def _inline_refs(schema):
    defs = schema.pop("$defs", {})

    def walk(x):
        if isinstance(x, dict):
            if "$ref" in x:
                return walk(dict(defs[x["$ref"].split("/")[-1]]))
            return {k: walk(v) for k, v in x.items() if k not in ("title",)}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x
    return walk(schema)


ARENA_TEXT = (
    "Arena (x east, y north, metres, 40 x 40 m, home pad at (0,-16)). Landmarks: "
    + "; ".join(f"{lm.name} = {lm.label} at ({lm.goal[0]:.0f},{lm.goal[1]:.0f})" for lm in arena.LANDMARKS)
    + ". The north side means landmarks with y > 0. A pond (no-fly zone) is at (2,-7)."
)

PLANNER_SYSTEM = (
    "You turn a drone operator's sentence into a mission plan. Drones: scout-1, scout-2 (scouts) and courier. "
    + ARENA_TEXT
    + " Rules: include only drones the operator mentions ('scouts' means both scouts). Use only the landmark names listed, or 'home'. "
    "Goals are in the order to fly them. Use action 'deliver' for the place a package goes, 'survey' when asked to map or explore, "
    "'visit' otherwise. When several drones share a survey area, split its landmarks between them. "
    "Put an 'avoid' constraint for every zone the operator says to avoid. return_home is true if the operator asks to come home or return. "
    "Do not add 'home' as a goal; use return_home instead."
)


def plan_mission(text, provider="sie", model=None):
    model = model or (llm.models()["mission_planner"] if provider == "sie" else llm.frontier_model())
    return llm.chat_json(provider, model, PLANNER_SYSTEM, f"Operator: {text}", plan_schema(), "mission_plan",
                         max_tokens=900, validate=lambda d: MissionPlan.model_validate(d))


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    landmark: Literal[tuple(LANDMARK_NAMES + ["none"])]
    bearing: Literal["ahead", "left", "right", "behind"]
    approx_distance: Literal["near", "medium", "far"]
    hazard: Literal["none", "water", "obstacle"]


PERCEPTION_SYSTEM = (
    "You read a drone's 360-degree panoramic camera frame (equirectangular: the centre column is straight ahead, "
    "the left quarter is to the drone's left, the right quarter to its right, both edges are behind). "
    "Report the single most prominent landmark. Landmarks: red_tower (tall red cylinder), blue_arch (blue gate of two posts and a beam), "
    "yellow_cubes (cluster of yellow cubes), green_tree_1..5 (green cylinders; pick the most prominent tree's number only if certain, else the "
    "nearest by the pose given). Use 'none' if no landmark is clearly visible. hazard: 'water' if the dark blue pond is visible below, "
    "'obstacle' if a landmark is very near and ahead, else 'none'."
)


def perceive(png, pose_text, provider="sie", model=None):
    model = model or (llm.models()["perception"] if provider == "sie" else llm.frontier_model())
    schema = _inline_refs(Observation.model_json_schema())
    return llm.chat_json(provider, model, PERCEPTION_SYSTEM, f"Drone pose: {pose_text}. Describe the frame.", schema, "observation",
                         image_png=png, max_tokens=200, validate=lambda d: Observation.model_validate(d))
