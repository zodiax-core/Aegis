"""Typed questions asked of Laya Vision. Laya encodes the image once and answers all questions in
one forward pass, so asking all of them costs little more than asking one."""
from __future__ import annotations

from typing import Dict, List

# key -> Laya question spec. 'noul' returns calibrated P(true); 'score' returns an ordinal level.
LAYA_QUESTIONS: Dict[str, dict] = {
    "physical_aggression": {"type": "noul", "instructions":
        "Does this scene show physical aggression between people, such as hitting, kicking or grappling?"},
    "weapon_visible": {"type": "noul", "instructions":
        "Is a weapon such as a knife or firearm visible in this scene?"},
    "vehicle_collision": {"type": "noul", "instructions":
        "Does this scene show a vehicle collision or a crashed vehicle?"},
    "fire_or_explosion": {"type": "noul", "instructions":
        "Does this scene show fire, smoke from a fire, or an explosion?"},
    "person_on_ground": {"type": "noul", "instructions":
        "Is a person lying on the ground or being forced to the ground?"},
    "property_damage": {"type": "noul", "instructions":
        "Does this scene show someone damaging or forcibly taking property?"},
    "crowd_panic": {"type": "noul", "instructions":
        "Are many people running or fleeing in panic?"},
    "severity": {"type": "score", "instructions": "How dangerous does this scene appear?",
                 "criteria": ["none: ordinary, safe scene", "mild: tense but no physical harm",
                              "moderate: physical altercation or clear hazard",
                              "severe: serious violence, injury or major hazard"]},
}
LAYA_KEYS: List[str] = list(LAYA_QUESTIONS.keys())
LAYA_FEAT_DIM = len(LAYA_KEYS)
SEVERITY_LEVELS = len(LAYA_QUESTIONS["severity"]["criteria"])
