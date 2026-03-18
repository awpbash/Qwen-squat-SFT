"""
Shared label constants for the VLM Squat Coach experiment pipeline.

All scripts import from here to avoid duplication and drift.
"""

# Individual label lists
STANCE_LABELS = ["shoulder-width", "narrow", "wide", "plie"]
DEPTH_LABELS = ["shallow", "90 degrees", "over 90 degrees"]
FORM_LABELS = ["back not straight", "knees over toes", "insufficient"]

# Canonical ordered list of all 12 labels
ALL_LABELS = STANCE_LABELS + DEPTH_LABELS + FORM_LABELS + ["hold", "not visible"]

# Label group sets (for aggregation)
GEOMETRIC_LABELS = set(STANCE_LABELS + DEPTH_LABELS)
HOLISTIC_LABELS = set(FORM_LABELS)
TEMPORAL_LABELS = {"hold"}

# Label groups dict (for oversampling and reporting)
LABEL_GROUPS = {
    "stance": list(STANCE_LABELS),
    "depth": list(DEPTH_LABELS),
    "form": list(FORM_LABELS),
    "variant": ["hold"],
    "meta": ["not visible"],
}

# Raw label prefix maps (raw QEVD format → structured)
STANCE_MAP = {
    "squats - shoulder-width": "shoulder-width",
    "squats - narrow": "narrow",
    "squats - wide": "wide",
    "squats - plie": "plie",
}
DEPTH_MAP = {
    "squats - shallow": "shallow",
    "squats - 90 degrees": "90 degrees",
    "squats - over 90 degrees": "over 90 degrees",
}
FORM_MAP = {
    "squats - back not straight": "back not straight",
    "squats - knees over toes": "knees over toes",
    "squats - insufficient": "insufficient",
}
