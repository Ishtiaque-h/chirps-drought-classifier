#!/usr/bin/env python
"""
Region registry for multi-region drought forecast experiments.

Bounding boxes are intentionally rectangular for the first multi-region path.
They are suitable for reproducible hydroclimate comparisons, but basin-polygon
masking should be added before making fine-grained regional claims.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    slug: str
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    rationale: str
    mask_countries: tuple[str, ...] = ()
    mask_note: str = ""

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Return bbox as (lat_min, lat_max, lon_min, lon_max)."""
        return (self.lat_min, self.lat_max, self.lon_min, self.lon_max)


REGIONS: dict[str, Region] = {
    "cvalley": Region(
        slug="cvalley",
        name="California Central Valley",
        lat_min=35.4,
        lat_max=40.6,
        lon_min=-122.5,
        lon_max=-119.0,
        rationale="Canonical project region; Mediterranean winter precipitation.",
        mask_countries=("United States of America",),
        mask_note=(
            "Country mask is only a land/country sanity check; it is not a "
            "Central Valley basin polygon."
        ),
    ),
    "southern_great_plains": Region(
        slug="southern_great_plains",
        name="Southern Great Plains",
        lat_min=33.0,
        lat_max=39.5,
        lon_min=-103.5,
        lon_max=-94.0,
        rationale=(
            "Contrasting continental regime with warm-season convective rainfall "
            "over Kansas, Oklahoma, and north Texas."
        ),
        mask_countries=("United States of America",),
        mask_note="Country mask should mainly confirm the rectangular box is within the U.S.",
    ),
    "murray_darling": Region(
        slug="murray_darling",
        name="Murray-Darling Basin bounding box",
        lat_min=-37.8,
        lat_max=-24.0,
        lon_min=138.0,
        lon_max=153.8,
        rationale=(
            "Semi-arid agricultural analogue with different ENSO teleconnection "
            "structure. This rectangular first pass should later be replaced by "
            "a basin polygon mask."
        ),
        mask_countries=("Australia",),
        mask_note=(
            "Country mask is not a Murray-Darling Basin polygon; a basin boundary "
            "is still needed for final claims."
        ),
    ),
    "mediterranean_spain": Region(
        slug="mediterranean_spain",
        name="Mediterranean Spain bounding box",
        lat_min=36.0,
        lat_max=42.8,
        lon_min=-6.5,
        lon_max=1.5,
        rationale=(
            "Mediterranean hydroclimate analogue spanning eastern/southern Spain; "
            "useful for testing whether the California result generalizes."
        ),
        mask_countries=("Spain",),
        mask_note="Spain mask checks whether the rectangular box includes non-Spain land cells.",
    ),
    "horn_of_africa": Region(
        slug="horn_of_africa",
        name="Horn of Africa bounding box",
        lat_min=-4.5,
        lat_max=13.5,
        lon_min=34.0,
        lon_max=43.0,
        rationale=(
            "Bimodal rainfall regime with stronger large-scale climate coupling; "
            "CHIRPS has strong heritage in this region."
        ),
        mask_countries=("Djibouti", "Eritrea", "Ethiopia", "Kenya", "Somalia"),
        mask_note=(
            "Country mask is a first-pass Horn land mask; it is not a hydrologic "
            "or livelihood-zone boundary."
        ),
    ),
}


ALIASES = {
    "central_valley": "cvalley",
    "california_central_valley": "cvalley",
    "great_plains": "southern_great_plains",
    "sgp": "southern_great_plains",
    "spain": "mediterranean_spain",
    "med_spain": "mediterranean_spain",
    "hoa": "horn_of_africa",
}


def resolve_region(name: str) -> Region:
    key = name.strip().lower().replace("-", "_")
    key = ALIASES.get(key, key)
    if key not in REGIONS:
        valid = ", ".join(sorted(REGIONS))
        raise KeyError(f"Unknown region '{name}'. Valid regions: {valid}")
    return REGIONS[key]


def region_table() -> str:
    rows = []
    for region in REGIONS.values():
        rows.append(
            f"{region.slug:<24} {region.name:<40} "
            f"lat[{region.lat_min:6.2f}, {region.lat_max:6.2f}] "
            f"lon[{region.lon_min:7.2f}, {region.lon_max:7.2f}]"
        )
    return "\n".join(rows)
