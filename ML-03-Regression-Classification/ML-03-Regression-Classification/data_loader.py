import os

import numpy as np
import pandas as pd

# The 28 skill columns used as features.
# "reactions" is left out on purpose: it correlates 0.86 with
# overall_rating because it is derived from it (data leakage).
SKILL_COLUMNS = [
    "crossing", "finishing", "heading_accuracy", "short_passing", "volleys",
    "dribbling", "curve", "freekick_accuracy", "long_passing", "ball_control",
    "acceleration", "sprint_speed", "agility", "balance", "shot_power",
    "jumping", "stamina", "strength", "long_shots", "aggression",
    "interceptions", "positioning", "vision", "penalties", "composure",
    "marking", "standing_tackle", "sliding_tackle",
]

META_COLUMNS = ["name", "age", "main_position", "role", "overall_rating"]


def load_data(csv_path, others_dir):
    """Read fifa_players_clean.csv. Returns (skills, meta)."""

    skills_path = os.path.join(others_dir, "skills.npy")
    meta_path = os.path.join(others_dir, "meta.csv")

    # Reuse the parsed copy if it exists
    if os.path.isfile(skills_path) and os.path.isfile(meta_path):
        print(f"Reused: {others_dir}")
        return np.load(skills_path), pd.read_csv(meta_path)

    df = pd.read_csv(csv_path)

    skills = df[SKILL_COLUMNS].to_numpy(dtype=np.uint8)
    meta = df[META_COLUMNS]

    os.makedirs(others_dir, exist_ok=True)
    np.save(skills_path, skills)
    meta.to_csv(meta_path, index=False)

    print(f"Loaded {len(skills)} rows from {os.path.basename(csv_path)}")

    return skills, meta


def to_features(skills):
    """(n, 28) uint8 0-100 -> float32 in 0-1."""

    return skills.astype(np.float32) / 100.0


def filter_roles(skills, meta, roles):
    """Keep only the given roles. Returns (skills, meta, index)."""

    mask = meta["role"].isin(roles).to_numpy()
    index = np.arange(len(skills))[mask]

    return skills[mask], meta[mask].reset_index(drop=True), index
