"""The locked FAIR1M-2.0 class order.

The list is the one the design document publishes (§2, five groups) and the
annotation files use verbatim — 11 aircraft, 9 ships, 10 vehicles, 4 fields, 3
road structures. It is fixed here rather than derived by sorting XML names so
that a class id in a run.json means the same thing across every worker, and so
that the `other-*` classes keep their published position instead of drifting to
the end (FAIR1M plan §4.5: "不能把不同类合并").

`background` is not a member: detection treats it as the softmax background slot
managed by the detector, so a 37-class model has 38 outputs internally.
"""

from __future__ import annotations

FAIR1M_CLASSES: tuple[str, ...] = (
    # aircraft (11)
    "Boeing737",
    "Boeing747",
    "Boeing777",
    "Boeing787",
    "ARJ21",
    "C919",
    "A220",
    "A321",
    "A330",
    "A350",
    "other-airplane",
    # ships (9)
    "Passenger Ship",
    "Motorboat",
    "Fishing Boat",
    "Tugboat",
    "Engineering Ship",
    "Liquid Cargo Ship",
    "Dry Cargo Ship",
    "Warship",
    "other-ship",
    # vehicles (10)
    "Small Car",
    "Bus",
    "Cargo Truck",
    "Dump Truck",
    "Van",
    "Trailer",
    "Tractor",
    "Truck Tractor",
    "Excavator",
    "other-vehicle",
    # fields (4)
    "Baseball Field",
    "Basketball Court",
    "Football Field",
    "Tennis Court",
    # road structures (3)
    "Roundabout",
    "Intersection",
    "Bridge",
)

FAIR1M_CLASS_TO_ID: dict[str, int] = {name: index for index, name in enumerate(FAIR1M_CLASSES)}
FAIR1M_ID_TO_CLASS: dict[int, str] = {index: name for name, index in FAIR1M_CLASS_TO_ID.items()}

# The five published groups, kept as id ranges so a report can print "vehicle
# AP" without re-deriving membership from the name.
FAIR1M_GROUPS: dict[str, range] = {
    "aircraft": range(0, 11),
    "ship": range(11, 20),
    "vehicle": range(20, 30),
    "field": range(30, 34),
    "road": range(34, 37),
}

# Spellings that appear in mirrors and in the reference repository's DIOR-borrowed
# loader. Mapped explicitly instead of being merged into a neighbour: an alias
# that has no target must fail loudly (see `class_id`).
FAIR1M_ALIASES: dict[str, str] = {
    "boeing737": "Boeing737",
    "boeing747": "Boeing747",
    "boeing777": "Boeing777",
    "boeing787": "Boeing787",
    "arj21": "ARJ21",
    "c919": "C919",
    "a220": "A220",
    "a321": "A321",
    "a330": "A330",
    "a350": "A350",
    "other-airplane": "other-airplane",
    "passenger ship": "Passenger Ship",
    "motorboat": "Motorboat",
    "fishing boat": "Fishing Boat",
    "tugboat": "Tugboat",
    "engineering ship": "Engineering Ship",
    "liquid cargo ship": "Liquid Cargo Ship",
    "dry cargo ship": "Dry Cargo Ship",
    "warship": "Warship",
    "other-ship": "other-ship",
    "small car": "Small Car",
    "bus": "Bus",
    "cargo truck": "Cargo Truck",
    "dump truck": "Dump Truck",
    "van": "Van",
    "trailer": "Trailer",
    "tractor": "Tractor",
    "truck tractor": "Truck Tractor",
    "excavator": "Excavator",
    "other-vehicle": "other-vehicle",
    "baseball field": "Baseball Field",
    "basketball court": "Basketball Court",
    "football field": "Football Field",
    "tennis court": "Tennis Court",
    "roundabout": "Roundabout",
    "intersection": "Intersection",
    "bridge": "Bridge",
}


class UnknownClassError(ValueError):
    """An annotation named a class that is not in the locked 37."""


def class_id(name: str) -> int:
    """Canonical id for an annotation class name (case-insensitive fallback)."""
    trimmed = name.strip()
    if trimmed in FAIR1M_CLASS_TO_ID:
        return FAIR1M_CLASS_TO_ID[trimmed]
    canonical = FAIR1M_ALIASES.get(trimmed.lower())
    if canonical is None:
        raise UnknownClassError(f"{name!r} is not one of the {len(FAIR1M_CLASSES)} FAIR1M classes")
    return FAIR1M_CLASS_TO_ID[canonical]
