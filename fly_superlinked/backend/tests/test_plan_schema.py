import pytest
from pydantic import ValidationError

from backend.agents.jobs import MissionPlan, plan_schema

GOOD = {"drones": [{"drone": "courier", "goals": [{"landmark": "red_tower", "action": "deliver"}],
                    "constraints": [{"type": "avoid", "zone": "pond"}], "return_home": True}]}


def test_valid_plan_parses():
    p = MissionPlan.model_validate(GOOD)
    assert p.drones[0].goals[0].landmark == "red_tower"


@pytest.mark.parametrize("bad", [
    {"drones": []},                                                                     # no drones
    {"drones": [{**GOOD["drones"][0], "drone": "scout-9"}]},                            # unknown drone
    {"drones": [{**GOOD["drones"][0], "goals": [{"landmark": "purple_castle", "action": "deliver"}]}]},  # invented landmark
    {"drones": [{**GOOD["drones"][0], "goals": [{"landmark": "red_tower", "action": "teleport"}]}]},
    {"drones": [{**GOOD["drones"][0], "extra": 1}]},                                   # extra field
    {"drones": [GOOD["drones"][0], GOOD["drones"][0]]},                                 # duplicate drone
    {"drones": [{k: v for k, v in GOOD["drones"][0].items() if k != "return_home"}]},  # missing field
])
def test_invalid_plans_rejected(bad):
    with pytest.raises(ValidationError):
        MissionPlan.model_validate(bad)


def test_json_schema_is_strict_and_self_contained():
    s = plan_schema()
    text = str(s)
    assert "$ref" not in text and "$defs" not in text
    assert s["additionalProperties"] is False
    drone = s["properties"]["drones"]["items"]
    assert drone["additionalProperties"] is False
    assert set(drone["required"]) == {"drone", "goals", "constraints", "return_home"}
