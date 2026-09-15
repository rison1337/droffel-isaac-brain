"""Shared annotation selectors; no random subsets or invented cell identities."""
import numpy as np


def make_groups(ann):
    types = ann["type"].fillna("").astype(str)
    sides = ann["somaSide"].fillna("").astype(str)
    classes = ann["class"].fillna("")
    photoreceptors = types.str.match(r"^R[1-6]($|[^0-9])")
    roots = ann["rootSide"].fillna(sides) if "rootSide" in ann else sides
    masks = {
        "loom_L": types.isin(["LC4", "LPLC2"]) & (sides == "L"),
        "loom_R": types.isin(["LC4", "LPLC2"]) & (sides == "R"),
        "GF": types == "DNp01",
        "steer_L": types.isin(["DNa01", "DNa02"]) & (sides == "L"),
        "steer_R": types.isin(["DNa01", "DNa02"]) & (sides == "R"),
        "flight": types.isin(["DNg02", "DNg07"]),
        "feed": types == "MN9",
        "odor": classes == "olfactory",
        "odor_a": types.isin(["ORN_DM1", "ORN_DM2", "ORN_DM4"]),
        "odor_b": types.isin(["ORN_DA1", "ORN_VA1d", "ORN_VA1v"]),
        "taste": classes == "gustatory",
        "touch": classes == "mechanosensory_tactile",
        "vibration": (classes == "mechanosensory") & types.str.startswith("JO"),
        "heat": classes == "thermosensory",
        "light": photoreceptors,
        "light_L": photoreceptors & (roots == "L"),
        "light_R": photoreceptors & (roots == "R"),
        "ALPN": classes == "ALPN",
        "KC": classes == "Kenyon_Cell",
        "MBON": classes == "MBON",
    }
    return {name: np.flatnonzero(mask.to_numpy()) for name, mask in masks.items()}


INPUT_HZ = {"loom_L": 350., "loom_R": 350., "odor": 35., "taste": 35., "light": 35.,
            "odor_a": 100., "odor_b": 100., "touch": 65., "vibration": 80., "heat": 100.,
            "light_L": 100., "light_R": 100.}
