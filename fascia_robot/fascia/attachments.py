from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttachmentZone:
    name: str
    body_name: str
    anatomical_region: str
    evidence_level: str = "moderate"


ZONE_BY_KEY = {
    "pelvis": AttachmentZone("pelvic_frame", "pelvis", "pelvis", "strong"),
    "torso": AttachmentZone("thoracic_frame", "torso_link", "thorax", "strong"),
    "left_shoulder": AttachmentZone("left_shoulder_frame", "left_shoulder_pitch_link", "left_shoulder", "strong"),
    "right_shoulder": AttachmentZone("right_shoulder_frame", "right_shoulder_pitch_link", "right_shoulder", "strong"),
    "left_elbow": AttachmentZone("left_forearm_origin", "left_elbow_link", "left_upper_limb", "moderate"),
    "right_elbow": AttachmentZone("right_forearm_origin", "right_elbow_link", "right_upper_limb", "moderate"),
    "left_wrist": AttachmentZone("left_distal_forearm", "left_wrist_roll_link", "left_forearm", "moderate"),
    "right_wrist": AttachmentZone("right_distal_forearm", "right_wrist_roll_link", "right_forearm", "moderate"),
    "left_hip": AttachmentZone("left_proximal_thigh", "left_hip_pitch_link", "left_thigh", "strong"),
    "right_hip": AttachmentZone("right_proximal_thigh", "right_hip_pitch_link", "right_thigh", "strong"),
    "left_knee": AttachmentZone("left_knee_adjacent", "left_knee_link", "left_knee", "moderate"),
    "right_knee": AttachmentZone("right_knee_adjacent", "right_knee_link", "right_knee", "moderate"),
    "left_ankle": AttachmentZone("left_ankle_heel", "left_ankle_pitch_link", "left_ankle_foot", "strong"),
    "right_ankle": AttachmentZone("right_ankle_heel", "right_ankle_pitch_link", "right_ankle_foot", "strong"),
}

