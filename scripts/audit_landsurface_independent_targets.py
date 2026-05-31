#!/usr/bin/env python
"""Audit independent soil-moisture target options for land-surface validation.

The current land-surface target is ERA5-Land root-zone dry fraction. This audit
does not re-score forecasts; it checks which independent or semi-independent
soil-moisture products would be scientifically suitable for external target
validation, which project regions they can cover, and whether matching local
files are already available.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
import shutil

import pandas as pd

from region_config import REGIONS
from run_gefsv12_landsurface_stack_benchmark import LANDSURFACE_DIR, PROJECT_ROOT


REPORT_DIR = PROJECT_ROOT / "results" / "report"
PAPER_DIR = REPORT_DIR / "paper"
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class TargetCandidate:
    product: str
    target_variable: str
    spatial_domain: str
    nominal_period: str
    independent_status: str
    validation_role: str
    covered_regions: tuple[str, ...]
    local_patterns: tuple[str, ...]
    source_url: str
    access_note: str
    priority: str


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-prefix", default="landsurface_independent_target_audit")
    parser.add_argument("--copy-report", action="store_true")
    return parser.parse_args()


def candidates() -> list[TargetCandidate]:
    all_regions = tuple(REGIONS)
    us_regions = ("cvalley", "southern_great_plains")
    return [
        TargetCandidate(
            product="NLDAS_NOAH0125_M",
            target_variable="Noah monthly soil moisture layers / top-1m soil moisture",
            spatial_domain="North America, 0.125 degree",
            nominal_period="1979-present monthly",
            independent_status="semi_independent_land_model",
            validation_role=(
                "Best immediate external check for U.S. regions because it covers the full "
                "hindcast/test period and is independent of ERA5-Land."
            ),
            covered_regions=us_regions,
            local_patterns=(
                "raw/nldas_noah_monthly/*/NLDAS_NOAH0125_M*.nc",
            ),
            source_url=(
                "https://catalog.data.gov/dataset/nldas-noah-land-surface-model-l4-monthly-"
                "0-125-x-0-125-degree-v2-0-nldas-noah0125-m-at-ges-66559"
            ),
            access_note="NASA GES DISC/Earthdata access; U.S./North America only.",
            priority="high_us_only",
        ),
        TargetCandidate(
            product="SMAP_L4_SPL4SM",
            target_variable="Surface and root-zone soil moisture",
            spatial_domain="Global 9 km EASE-Grid",
            nominal_period="2015-present, 3-hourly",
            independent_status="satellite_assimilated_land_model",
            validation_role=(
                "Most defensible global root-zone validation product, but the record is short "
                "and mainly supports 2017-2019 test validation, not the full calibration window."
            ),
            covered_regions=all_regions,
            local_patterns=(
                "raw/smap_l4_spl4smgp*/subsets/*/*/SMAP_L4_SM_gph_*__*__sm_rootzone_pctl*.nc4",
                "processed/smap_l4_rootzone_pctl_monthly_*_2015_2019.nc",
            ),
            source_url="https://catalog.data.gov/dataset/smap-l4-global-3-hourly-9-km-ease-grid-surface-and-root-zone-soil-moisture-geophysical-dat-35042",
            access_note="NASA NSIDC/Earthdata access; requires reprojection or spatial averaging from EASE-Grid.",
            priority="high_global_short_record",
        ),
        TargetCandidate(
            product="GLDAS_NOAH025_M",
            target_variable="Noah root-zone soil moisture / soil moisture layers",
            spatial_domain="Global 0.25 degree",
            nominal_period="2000-present monthly for GLDAS-2.1",
            independent_status="semi_independent_land_model",
            validation_role=(
                "Useful global model-product sensitivity check; less independent than SMAP or "
                "ESA CCI because it is another land-surface model product."
            ),
            covered_regions=all_regions,
            local_patterns=(
                "raw/gldas_noah_monthly/*/GLDAS_NOAH025_M*.nc4",
            ),
            source_url="https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_NOAH025_M.2.1/doc/README_GLDAS2.pdf",
            access_note="NASA GES DISC/Earthdata access; global but model-based.",
            priority="medium_global_model_sensitivity",
        ),
        TargetCandidate(
            product="ESA_CCI_SM_COMBINED",
            target_variable="Surface soil moisture",
            spatial_domain="Global 0.25 degree",
            nominal_period="Long-term daily climate data record",
            independent_status="satellite_surface_retrieval",
            validation_role=(
                "Good independent surface-moisture direction-of-change check; target mismatch "
                "is substantial because it does not measure root-zone moisture."
            ),
            covered_regions=all_regions,
            local_patterns=(
                "*esa*cci*soil*moisture*.nc",
                "*cci*sm*.nc",
                "*ESACCI*SOILMOISTURE*.nc",
            ),
            source_url="https://climate.esa.int/en/projects/soil-moisture/",
            access_note="Surface-only product; use as sensitivity/consistency check, not root-zone replacement.",
            priority="medium_surface_only",
        ),
        TargetCandidate(
            product="GLEAM_ROOTZONE_SM",
            target_variable="Root-zone soil moisture",
            spatial_domain="Global",
            nominal_period="Multi-decadal monthly/daily, version-dependent",
            independent_status="hydrological_model_reanalysis",
            validation_role=(
                "Useful global hydrological sensitivity target if access/licensing is satisfied; "
                "should be reported as model-product validation rather than direct observation."
            ),
            covered_regions=all_regions,
            local_patterns=(
                "*gleam*root*zone*.nc",
                "*gleam*rzsm*.nc",
                "*GLEAM*.nc",
            ),
            source_url="https://www.gleam.eu/",
            access_note="Registration/licensing and variable availability need confirmation before use.",
            priority="medium_global_model_sensitivity",
        ),
    ]


def find_local_files(data_dir: Path, patterns: tuple[str, ...]) -> list[str]:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(data_dir.rglob(pattern))
    unique = sorted({path.resolve() for path in matches if path.is_file()})
    return [str(path.relative_to(PROJECT_ROOT)) for path in unique]


def build_audit(args: Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in candidates():
        local_files = find_local_files(args.data_dir, candidate.local_patterns)
        rows.append(
            {
                "product": candidate.product,
                "target_variable": candidate.target_variable,
                "spatial_domain": candidate.spatial_domain,
                "nominal_period": candidate.nominal_period,
                "independent_status": candidate.independent_status,
                "validation_role": candidate.validation_role,
                "covered_regions": " ".join(candidate.covered_regions),
                "n_project_regions_covered": len(candidate.covered_regions),
                "local_file_count": len(local_files),
                "local_files": " ".join(local_files[:10]),
                "ready_to_score_now": bool(local_files),
                "source_url": candidate.source_url,
                "access_note": candidate.access_note,
                "priority": candidate.priority,
                "current_decision": current_decision(candidate, bool(local_files)),
            }
        )
    df = pd.DataFrame(rows)
    priority_rank = {
        "high_us_only": 0,
        "high_global_short_record": 1,
        "medium_global_model_sensitivity": 2,
        "medium_surface_only": 3,
    }
    df["_priority_rank"] = df["priority"].map(priority_rank).fillna(99).astype(int)
    return (
        df.sort_values(["ready_to_score_now", "_priority_rank", "product"], ascending=[False, True, True])
        .drop(columns=["_priority_rank"])
        .reset_index(drop=True)
    )


def current_decision(candidate: TargetCandidate, ready: bool) -> str:
    if (
        ready
        and candidate.product == "NLDAS_NOAH0125_M"
        and (REPORT_DIR / "landsurface" / "landsurface_nldas_gefsv12_validation_summary.csv").exists()
    ):
        return (
            "Local files are present and the two U.S. region GEFSv12 validation has been scored; "
            "use table18_landsurface_nldas_validation.csv as the manuscript-facing result."
        )
    if (
        ready
        and candidate.product == "GLDAS_NOAH025_M"
        and (REPORT_DIR / "landsurface" / "landsurface_gldas_gefsv12_validation_summary.csv").exists()
    ):
        return (
            "Local files are present and the five-region GEFSv12 validation has been scored; "
            "use the GLDAS validation and ERA5-Land/GLDAS comparison tables as manuscript-facing sensitivity results."
        )
    if (
        ready
        and candidate.product == "SMAP_L4_SPL4SM"
        and (REPORT_DIR / "landsurface" / "landsurface_smap_l4_midmonth_gefsv12_validation_summary.csv").exists()
    ):
        return (
            "Local regional subsets are present and the five-region short-record GEFSv12 validation has been scored; "
            "use table22_landsurface_smap_l4_validation.csv as satellite-assimilated target validation with short-record caveats."
        )
    if ready:
        return "Local files are present; next step is to harmonize to monthly regional dry fractions and rescore."
    if candidate.product == "NLDAS_NOAH0125_M":
        return (
            "Acquire first for Central Valley and Southern Great Plains independent U.S. validation "
            "using scripts/download_nldas_noah_monthly.py, then score with scripts/run_nldas_landsurface_validation.py."
        )
    if candidate.product == "SMAP_L4_SPL4SM":
        return "Acquire if global independent validation is needed; expect short-record validation only."
    if candidate.product == "GLDAS_NOAH025_M":
        return "Acquire only as a global model-product sensitivity check, not as independent satellite validation."
    if candidate.product == "ESA_CCI_SM_COMBINED":
        return "Use only as surface-moisture consistency check, not as the primary root-zone target."
    return "Keep as optional sensitivity target after access/licensing is confirmed."


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Independent Land-Surface Target Audit",
        "",
        "This table records whether an external soil-moisture target is ready for validation. It intentionally separates true validation from feasible next acquisition steps.",
        "",
    ]
    ready = df.loc[df["ready_to_score_now"]]
    if ready.empty:
        lines.append("No independent soil-moisture target files are currently present under `data/`; the current positive land-surface result remains ERA5-Land-targeted until one of these products is acquired and harmonized.")
    else:
        lines.append(
            "Local independent-target files are present for: "
            + ", ".join(ready["product"].astype(str).tolist())
            + "."
        )
        lines.extend(
            [
                "",
                "Completed checks: NLDAS for the two U.S. regions, GLDAS as a five-region model-product sensitivity, and SMAP L4 as a five-region short-record satellite-assimilated target check.",
                "Recommended remaining order: GLEAM/ESA CCI only as optional sensitivity checks; prioritize manuscript synthesis before adding more target products.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    LANDSURFACE_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    audit = build_audit(args)
    out_csv = out_dir / f"{args.out_prefix}.csv"
    out_md = out_dir / f"{args.out_prefix}.md"
    audit.to_csv(out_csv, index=False)
    write_markdown(audit, out_md)

    if args.copy_report:
        shutil.copy2(out_csv, LANDSURFACE_DIR / out_csv.name)
        shutil.copy2(out_md, LANDSURFACE_DIR / out_md.name)
        shutil.copy2(out_csv, PAPER_DIR / "table17_landsurface_independent_target_audit.csv")

    print(out_md.read_text(encoding="utf-8"))
    print(f"Wrote independent-target audit: {out_csv} rows={len(audit)}")
    if args.copy_report:
        print(f"Copied paper table: {PAPER_DIR / 'table17_landsurface_independent_target_audit.csv'}")


if __name__ == "__main__":
    main()
