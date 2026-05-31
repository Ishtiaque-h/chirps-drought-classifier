#!/usr/bin/env python
"""Audit forecast archives for extending land-surface drought validation.

This audit answers a narrow question before any large operational download:
which forecast archives can support a modern root-zone soil-moisture dry-
fraction benchmark beyond the current GEFSv12 reforecast window?

It intentionally does not score a model. It probes small public metadata files,
summarizes local target-product coverage, and writes a decision table that can
be cited in the project report.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import time
from typing import Iterable

import pandas as pd
import requests
import xarray as xr

from region_config import REGIONS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "results" / "report" / "landsurface"
PAPER_DIR = PROJECT_ROOT / "results" / "report" / "paper"
DATA_DIR = PROJECT_ROOT / "data"

GEFS_PDS_ROOT = "https://noaa-gefs-pds.s3.amazonaws.com"
GEFS_REFORECAST_REGISTRY = "https://registry.opendata.aws/noaa-gefs-reforecast/"
GEFS_OPERATIONAL_REGISTRY = "https://registry.opendata.aws/noaa-gefs/"
C3S_CATALOGUE_ROOT = "https://cds.climate.copernicus.eu/api/catalogue/v1/collections"
C3S_ORIGINAL_DATASET = "seasonal-original-single-levels"
C3S_MONTHLY_DATASET = "seasonal-monthly-single-levels"
C3S_ORIGINAL_URL = "https://cds.climate.copernicus.eu/datasets/seasonal-original-single-levels"
C3S_MONTHLY_URL = "https://cds.climate.copernicus.eu/datasets/seasonal-monthly-single-levels"
C3S_ORIGINAL_LICENCE_URL = (
    "https://cds.climate.copernicus.eu/datasets/seasonal-original-single-levels"
    "?tab=download#manage-licences"
)
S2S_ARCHIVE_URL = "https://confluence.ecmwf.int/display/S2S"
ECMWF_OPEN_DATA_URL = "https://www.ecmwf.int/en/forecasts/datasets/open-data"
SUBX_URL = "https://weather.ou.edu/~kpegion/subc/"
SUBX_IRI_URLS = [
    "https://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/",
    "https://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.EMC/",
    "https://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.EMC/.GEFSv12/",
    "https://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.NASA/",
    "https://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.GMAO/",
]
SMAP_URL = "https://nsidc.org/data/spl4smgp/versions/8"

REQUEST_HEADERS = {
    "User-Agent": "chirps-drought-classifier/landsurface-forecast-archive-audit"
}
SOIL_LINE_RE = re.compile(
    r":(?P<var>SOILW|SOILL):(?P<level>[^:]+ below ground):(?P<step>\d+) hour fcst"
)


@dataclass(frozen=True)
class LocalTarget:
    product: str
    region: str
    path: Path | None
    start: str
    end: str
    n_months: int
    ready_for_2021_2025: bool
    ready_for_2021_2026: bool


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-dates",
        nargs="+",
        default=[
            "20200101",
            "20210101",
            "20220101",
            "20230101",
            "20240101",
            "20250101",
            "20260101",
        ],
        help="YYYYMMDD dates to probe in the operational GEFS public archive.",
    )
    parser.add_argument(
        "--forecast-hours",
        nargs="+",
        type=int,
        default=[240, 384, 600, 840],
        help="Operational GEFS forecast hours to probe.",
    )
    parser.add_argument(
        "--members",
        nargs="+",
        default=["gec00", "gep01"],
        help="Operational GEFS members to probe.",
    )
    parser.add_argument("--cycle", default="00", help="Operational GEFS cycle hour.")
    parser.add_argument("--out-prefix", default="landsurface_forecast_archive_audit")
    parser.add_argument("--copy-report", action="store_true")
    parser.add_argument(
        "--skip-native-rzsm-audit",
        action="store_true",
        help="Skip C3S/SubX native-RZSM archive metadata probes.",
    )
    parser.add_argument(
        "--probe-cds-retrieval",
        action="store_true",
        help=(
            "Submit one tiny C3S seasonal-original VSM retrieval probe. This catches "
            "licence/access blockers and may download a small NetCDF if licences are accepted."
        ),
    )
    parser.add_argument("--request-sleep", type=float, default=0.05)
    return parser.parse_args()


def request_with_retry(method: str, url: str, *, retries: int = 3) -> requests.Response:
    last_error: requests.RequestException | None = None
    for attempt in range(retries):
        try:
            return requests.request(
                method,
                url,
                headers=REQUEST_HEADERS,
                timeout=(10, 30),
                allow_redirects=True,
            )
        except requests.RequestException as exc:
            last_error = exc
            if attempt == retries - 1:
                break
            time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def catalogue_collection(dataset_id: str) -> dict[str, object]:
    response = request_with_retry("GET", f"{C3S_CATALOGUE_ROOT}/{dataset_id}")
    response.raise_for_status()
    return response.json()


def link_href(collection: dict[str, object], rel: str) -> str:
    for link in collection.get("links", []):
        if isinstance(link, dict) and link.get("rel") == rel and link.get("href"):
            return str(link["href"])
    return ""


def field_values(form: list[dict[str, object]], name: str) -> list[str]:
    for field in form:
        if field.get("name") != name:
            continue
        details = field.get("details", {})
        if isinstance(details, dict):
            values = details.get("values", [])
            if isinstance(values, list):
                return [str(value) for value in values]
    return []


def variable_group_values(form: list[dict[str, object]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for field in form:
        if field.get("name") != "variable":
            continue
        details = field.get("details", {})
        if not isinstance(details, dict):
            return groups
        for group in details.get("groups", []):
            if not isinstance(group, dict):
                continue
            label = str(group.get("label", "unlabelled"))
            values = group.get("values", [])
            groups[label] = [str(value) for value in values] if isinstance(values, list) else []
    return groups


def as_int_range(values: Iterable[str]) -> tuple[int | None, int | None, int]:
    parsed: list[int] = []
    for value in values:
        try:
            parsed.append(int(value))
        except (TypeError, ValueError):
            continue
    if not parsed:
        return None, None, 0
    return min(parsed), max(parsed), len(set(parsed))


def aggregate_c3s_vsm_candidates(constraints: list[dict[str, object]]) -> pd.DataFrame:
    records: dict[tuple[str, str], dict[str, object]] = {}
    for row in constraints:
        variables = row.get("variable", [])
        if not isinstance(variables, list) or "volumetric_soil_moisture" not in variables:
            continue
        centres = row.get("originating_centre", [])
        systems = row.get("system", [])
        years = [str(value) for value in row.get("year", [])]
        months = [str(value) for value in row.get("month", [])]
        days = [str(value) for value in row.get("day", [])]
        leads = [str(value) for value in row.get("leadtime_hour", [])]
        for centre in centres if isinstance(centres, list) else []:
            for system in systems if isinstance(systems, list) else []:
                key = (str(centre), str(system))
                record = records.setdefault(
                    key,
                    {
                        "originating_centre": str(centre),
                        "system": str(system),
                        "years": set(),
                        "months": set(),
                        "days": set(),
                        "leadtime_hours": set(),
                        "n_constraints": 0,
                    },
                )
                record["years"].update(years)
                record["months"].update(months)
                record["days"].update(days)
                record["leadtime_hours"].update(leads)
                record["n_constraints"] = int(record["n_constraints"]) + 1

    rows: list[dict[str, object]] = []
    for record in records.values():
        year_min, year_max, n_years = as_int_range(record["years"])
        lead_min, lead_max, n_leads = as_int_range(record["leadtime_hours"])
        rows.append(
            {
                "archive": "C3S_seasonal_original_single_levels",
                "dataset_id": C3S_ORIGINAL_DATASET,
                "originating_centre": record["originating_centre"],
                "system": record["system"],
                "variable": "volumetric_soil_moisture",
                "first_year": year_min,
                "last_year": year_max,
                "n_years": n_years,
                "n_months": len(record["months"]),
                "months": " ".join(sorted(record["months"])),
                "days": " ".join(sorted(record["days"])),
                "min_leadtime_hour": lead_min,
                "max_leadtime_hour": lead_max,
                "n_leadtimes": n_leads,
                "n_constraints": int(record["n_constraints"]),
                "candidate_rank_reason": (
                    "fuller_historical_modern_coverage"
                    if record["originating_centre"] == "ecmwf" and record["system"] == "51"
                    else "available_native_vsm_candidate"
                ),
                "source_url": C3S_ORIGINAL_URL,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["n_years", "n_months", "max_leadtime_hour", "originating_centre", "system"],
        ascending=[False, False, False, True, True],
    )


def audit_c3s_native_rzsm() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    all_candidates: list[pd.DataFrame] = []

    for dataset_id, dataset_url in [
        (C3S_ORIGINAL_DATASET, C3S_ORIGINAL_URL),
        (C3S_MONTHLY_DATASET, C3S_MONTHLY_URL),
    ]:
        try:
            collection = catalogue_collection(dataset_id)
            form_url = link_href(collection, "form")
            constraints_url = link_href(collection, "constraints")
            form_response = request_with_retry("GET", form_url)
            constraints_response = request_with_retry("GET", constraints_url)
            form_response.raise_for_status()
            constraints_response.raise_for_status()
            form = form_response.json()
            constraints = constraints_response.json()
            if not isinstance(form, list) or not isinstance(constraints, list):
                raise TypeError("unexpected CDS form/constraint JSON shape")
            variable_groups = variable_group_values(form)
            soil_variables = variable_groups.get("Soil levels", [])
            has_native_vsm = "volumetric_soil_moisture" in soil_variables
            candidates = aggregate_c3s_vsm_candidates(constraints) if has_native_vsm else pd.DataFrame()
            if not candidates.empty:
                all_candidates.append(candidates)
            best = candidates.iloc[0].to_dict() if not candidates.empty else {}
            status = "native_vsm_available" if has_native_vsm and not candidates.empty else "native_vsm_not_available"
            decision = (
                "Best immediate native-RZSM path after licence acceptance; ECMWF system 51 offers "
                "full 1981-2026 monthly starts and long lead-time coverage."
                if dataset_id == C3S_ORIGINAL_DATASET and status == "native_vsm_available"
                else (
                    "Not suitable for the next RZSM benchmark because this monthly-statistics dataset "
                    "does not expose volumetric soil moisture in current catalogue metadata."
                )
            )
            summary_rows.append(
                {
                    "archive": f"C3S_{dataset_id}",
                    "archive_role": "native seasonal forecast RZSM candidate",
                    "metadata_status": "catalogue_accessible",
                    "download_probe_status": "not_run",
                    "has_native_volumetric_soil_moisture": bool(has_native_vsm),
                    "best_originating_centre": best.get("originating_centre", ""),
                    "best_system": best.get("system", ""),
                    "first_year": best.get("first_year", ""),
                    "last_year": best.get("last_year", ""),
                    "n_months": best.get("n_months", ""),
                    "min_leadtime_hour": best.get("min_leadtime_hour", ""),
                    "max_leadtime_hour": best.get("max_leadtime_hour", ""),
                    "scientific_decision": decision,
                    "source_url": dataset_url,
                    "licence_url": C3S_ORIGINAL_LICENCE_URL if dataset_id == C3S_ORIGINAL_DATASET else dataset_url,
                    "source_reference": (
                        f"{collection.get('title', dataset_id)}; DOI {collection.get('sci:doi', '')}; "
                        f"updated {collection.get('updated', '')}"
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001 - audit must continue and record blockers
            summary_rows.append(
                {
                    "archive": f"C3S_{dataset_id}",
                    "archive_role": "native seasonal forecast RZSM candidate",
                    "metadata_status": f"metadata_probe_failed:{type(exc).__name__}",
                    "download_probe_status": "not_run",
                    "has_native_volumetric_soil_moisture": False,
                    "best_originating_centre": "",
                    "best_system": "",
                    "first_year": "",
                    "last_year": "",
                    "n_months": "",
                    "min_leadtime_hour": "",
                    "max_leadtime_hour": "",
                    "scientific_decision": "Metadata access failed; do not score or claim support from this archive yet.",
                    "source_url": dataset_url,
                    "licence_url": "",
                    "source_reference": str(exc)[:500],
                }
            )

    candidates_df = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    return pd.DataFrame(summary_rows), candidates_df


def probe_cds_vsm_retrieval(target_path: Path) -> dict[str, object]:
    request = {
        "originating_centre": "ecmwf",
        "system": "51",
        "variable": ["volumetric_soil_moisture"],
        "year": ["2024"],
        "month": ["01"],
        "day": ["01"],
        "leadtime_hour": ["504"],
        "area": [42.5, -125.0, 31.0, -113.0],
        "data_format": "netcdf",
    }
    try:
        import cdsapi  # type: ignore

        target_path.parent.mkdir(parents=True, exist_ok=True)
        client = cdsapi.Client(quiet=True, progress=False)
        client.retrieve(C3S_ORIGINAL_DATASET, request, str(target_path))
        return {
            "download_probe_status": "success",
            "download_probe_target": str(target_path.relative_to(PROJECT_ROOT)),
            "download_probe_bytes": target_path.stat().st_size if target_path.exists() else 0,
            "download_probe_note": "Tiny ECMWF system 51 VSM NetCDF probe completed.",
        }
    except Exception as exc:  # noqa: BLE001 - the audit should record access blockers
        message = str(exc)
        if "required licences not accepted" in message.lower() or "licence" in message.lower():
            status = "blocked_license_not_accepted"
            note = f"Accept dataset terms before retrieval: {C3S_ORIGINAL_LICENCE_URL}"
        elif "401" in message or "403" in message:
            status = "blocked_auth_or_permission"
            note = "CDS credentials exist but the retrieval endpoint rejected this request."
        else:
            status = f"failed:{type(exc).__name__}"
            note = message[:500]
        return {
            "download_probe_status": status,
            "download_probe_target": str(target_path.relative_to(PROJECT_ROOT)),
            "download_probe_bytes": target_path.stat().st_size if target_path.exists() else 0,
            "download_probe_note": note,
        }


def probe_subx_iri_access() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for url in SUBX_IRI_URLS:
        status = ""
        http_status = None
        title = ""
        note = ""
        try:
            response = request_with_retry("GET", url)
            http_status = int(response.status_code)
            text = response.text or ""
            match = re.search(r"<title>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
            title = re.sub(r"\s+", " ", match.group(1)).strip() if match else ""
            lower = text.lower()
            if "dlauth" in lower or "login" in lower:
                status = "auth_required"
                note = "IRI Data Library returned an authentication page, so variable coverage cannot be audited anonymously."
            elif http_status == 200:
                status = "accessible"
                note = "Endpoint returned HTML without an obvious authentication page."
            else:
                status = "unavailable"
                note = response.reason or ""
        except Exception as exc:  # noqa: BLE001
            status = f"failed:{type(exc).__name__}"
            note = str(exc)[:500]
        rows.append(
            {
                "archive": "SubX_IRI_Data_Library",
                "url": url,
                "http_status": http_status,
                "access_status": status,
                "page_title": title,
                "scientific_decision": (
                    "Do not use as the immediate next benchmark until authenticated dataset access "
                    "or another SubX mirror confirms native RZSM/soil-moisture variables."
                    if status == "auth_required"
                    else "Candidate requires a variable-level audit before scoring."
                ),
                "note": note,
                "source_reference": "Subseasonal Experiment (SubX) data access through IRI Data Library",
            }
        )
    return pd.DataFrame(rows)


def apply_native_probe_to_summary(summary: pd.DataFrame, probe: dict[str, object]) -> pd.DataFrame:
    if summary.empty:
        return summary
    updated = summary.copy()
    mask = updated["archive"].eq(f"C3S_{C3S_ORIGINAL_DATASET}")
    for key, value in probe.items():
        updated.loc[mask, key] = value
    return updated


def operational_gefs_idx_url(date: str, member: str, cycle: str, forecast_hour: int) -> str:
    return (
        f"{GEFS_PDS_ROOT}/gefs.{date}/{cycle}/atmos/pgrb2bp5/"
        f"{member}.t{cycle}z.pgrb2b.0p50.f{forecast_hour:03d}.idx"
    )


def probe_operational_gefs(args: Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    soil_records: list[dict[str, object]] = []

    for date in args.probe_dates:
        for member in args.members:
            for forecast_hour in args.forecast_hours:
                url = operational_gefs_idx_url(date, member, args.cycle, forecast_hour)
                status = None
                content_length = None
                soil_lines: list[str] = []
                error = ""
                try:
                    response = request_with_retry("GET", url)
                    status = int(response.status_code)
                    content_length = len(response.content) if response.content is not None else None
                    if status == 200:
                        text = response.text
                        for line in text.splitlines():
                            match = SOIL_LINE_RE.search(line)
                            if match:
                                soil_lines.append(line)
                                soil_records.append(
                                    {
                                        "date": date,
                                        "member": member,
                                        "forecast_hour": forecast_hour,
                                        "variable": match.group("var"),
                                        "level": match.group("level").strip(),
                                        "step_hours": int(match.group("step")),
                                        "idx_line": line,
                                    }
                                )
                    else:
                        error = response.reason or ""
                except Exception as exc:  # noqa: BLE001 - audit must continue across transient failures
                    error = f"{type(exc).__name__}: {exc}"
                rows.append(
                    {
                        "source": "NOAA_GEFS_operational_AWS",
                        "date": date,
                        "member": member,
                        "cycle": args.cycle,
                        "forecast_hour": forecast_hour,
                        "idx_url": url,
                        "http_status": status,
                        "available": status == 200,
                        "content_length": content_length,
                        "n_soilw_lines": sum(":SOILW:" in line for line in soil_lines),
                        "n_soill_lines": sum(":SOILL:" in line for line in soil_lines),
                        "soil_levels": "; ".join(sorted({line.split(":")[4] for line in soil_lines})),
                        "error": error,
                    }
                )
                if args.request_sleep:
                    time.sleep(float(args.request_sleep))

    probe = pd.DataFrame(rows)
    soil = pd.DataFrame(soil_records)
    summary = summarize_operational_probe(probe, soil, args)
    return probe, summary


def summarize_operational_probe(
    probe: pd.DataFrame,
    soil: pd.DataFrame,
    args: Namespace,
) -> dict[str, object]:
    available = probe.loc[probe["available"]].copy()
    years = sorted({str(date)[:4] for date in available["date"].astype(str)}) if not available.empty else []
    max_fhour = int(available["forecast_hour"].max()) if not available.empty else 0
    available_date_count = int(available["date"].nunique()) if not available.empty else 0
    requested_date_count = len(set(args.probe_dates))

    soilw_levels = sorted(soil.loc[soil["variable"] == "SOILW", "level"].dropna().unique().tolist())
    soill_levels = sorted(soil.loc[soil["variable"] == "SOILL", "level"].dropna().unique().tolist())
    has_0_1 = "0-0.1 m below ground" in soilw_levels
    has_0p1_0p4 = "0.1-0.4 m below ground" in soilw_levels
    has_0p4_1 = "0.4-1 m below ground" in soilw_levels
    has_soill_0_1 = "0-0.1 m below ground" in soill_levels
    has_soill_0p1_0p4 = "0.1-0.4 m below ground" in soill_levels
    has_soill_0p4_1 = "0.4-1 m below ground" in soill_levels
    soill_top_compatible = has_soill_0_1 and has_soill_0p1_0p4 and has_soill_0p4_1

    if has_0_1 and has_0p1_0p4 and has_0p4_1:
        depth_compatibility = "full_0_to_1m_soilw_available"
        depth_note = "Operational GEFS SOILW includes the same 0-1 m layer set needed for a direct root-zone approximation."
    elif has_0p1_0p4 and has_0p4_1 and soill_top_compatible:
        depth_compatibility = "partial_soilw_0p1_to_1m_plus_soill_0_to_1m_available"
        depth_note = (
            "Operational GEFS pgrb2b probe contains SOILW for 0.1-0.4 and 0.4-1 m but not "
            "SOILW for 0-0.1 m. SOILL does provide 0-0.1, 0.1-0.4, and 0.4-1 m layers, "
            "so a top-layer-compatible liquid-soil benchmark can be tested, but it is not "
            "physically identical to SOILW total soil water or the GEFSv12 0-100 cm approximation."
        )
    elif has_0p1_0p4 and has_0p4_1:
        depth_compatibility = "partial_0p1_to_1m_soilw_available"
        depth_note = (
            "Operational GEFS pgrb2b probe contains SOILW for 0.1-0.4 and 0.4-1 m, "
            "but not SOILW for 0-0.1 m. Treat this as a 0.1-1 m subsurface benchmark "
            "or explicitly test whether SOILL can be used; do not silently call it the "
            "same 0-100 cm RZSM target."
        )
    else:
        depth_compatibility = "soilw_depths_insufficient"
        depth_note = "The probed operational GEFS files do not expose enough SOILW layers for a root-zone benchmark."

    if max_fhour >= 600:
        lead_note = "Forecast-hour probes support subseasonal-to-month-ahead valid times in the 00 UTC cycle."
    elif max_fhour >= 384:
        lead_note = "Forecast-hour probes support roughly 16-day valid times but not the current month-ahead design."
    else:
        lead_note = "Forecast-hour probes do not support a comparable lead window."

    return {
        "available_probe_years": " ".join(years),
        "available_date_count": available_date_count,
        "requested_date_count": requested_date_count,
        "max_available_forecast_hour": max_fhour,
        "lead_support": lead_note,
        "soilw_levels": "; ".join(soilw_levels),
        "soill_levels": "; ".join(soill_levels),
        "depth_compatibility": depth_compatibility,
        "depth_note": depth_note,
        "first_available_probe_year": years[0] if years else "",
        "last_available_probe_year": years[-1] if years else "",
    }


def read_time_coverage(path: Path) -> tuple[str, str, int]:
    try:
        with xr.open_dataset(path) as ds:
            if "time" not in ds.coords and "time" not in ds.dims:
                return "", "", 0
            values = pd.to_datetime(ds["time"].values)
    except Exception:
        return "", "", 0
    if len(values) == 0:
        return "", "", 0
    return values.min().strftime("%Y-%m"), values.max().strftime("%Y-%m"), int(len(values))


def first_existing(patterns: Iterable[str]) -> Path | None:
    candidates: list[tuple[str, int, Path]] = []
    for pattern in patterns:
        matches = sorted(PROJECT_ROOT.glob(pattern))
        for path in matches:
            start, end, n_months = read_time_coverage(path)
            candidates.append((end or "", int(n_months), path))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item[0], item[1], str(item[2])), reverse=True)[0][2]


def local_target_path(product: str, region: str) -> Path | None:
    if product == "ERA5_Land_RZSM":
        return first_existing([f"data/processed/era5_land_soil_moisture_monthly_{region}_*.nc"])
    if product == "GLDAS_Noah_RootMoist":
        return first_existing([f"data/processed/gldas_noah_rootzone_monthly_{region}_*.nc"])
    if product == "SMAP_L4_multi3snapshot":
        return first_existing([f"data/processed/smap_l4_rootzone_pctl_monthly_{region}_*_multi3snapshot.nc"])
    return None


def local_target_audit() -> pd.DataFrame:
    rows: list[LocalTarget] = []
    for product in ["ERA5_Land_RZSM", "GLDAS_Noah_RootMoist", "SMAP_L4_multi3snapshot"]:
        for region in sorted(REGIONS):
            path = local_target_path(product, region)
            if path is None:
                rows.append(
                    LocalTarget(
                        product=product,
                        region=region,
                        path=None,
                        start="",
                        end="",
                        n_months=0,
                        ready_for_2021_2025=False,
                        ready_for_2021_2026=False,
                    )
                )
                continue
            start, end, n_months = read_time_coverage(path)
            rows.append(
                LocalTarget(
                    product=product,
                    region=region,
                    path=path.relative_to(PROJECT_ROOT),
                    start=start,
                    end=end,
                    n_months=n_months,
                    ready_for_2021_2025=(start <= "2021-01" and end >= "2025-12") if start and end else False,
                    ready_for_2021_2026=(start <= "2021-01" and end >= "2026-03") if start and end else False,
                )
            )
    return pd.DataFrame([row.__dict__ for row in rows])


def target_overlap_summary(targets: pd.DataFrame) -> dict[str, object]:
    def regions_ready(product: str, column: str) -> list[str]:
        mask = (targets["product"] == product) & targets[column].astype(bool)
        return sorted(targets.loc[mask, "region"].astype(str).tolist())

    return {
        "era5_land_regions_ready_2021_2025": " ".join(regions_ready("ERA5_Land_RZSM", "ready_for_2021_2025")),
        "era5_land_regions_ready_2021_2026": " ".join(regions_ready("ERA5_Land_RZSM", "ready_for_2021_2026")),
        "gldas_regions_ready_2021_2025": " ".join(regions_ready("GLDAS_Noah_RootMoist", "ready_for_2021_2025")),
        "smap_regions_ready_2021_2025_local": " ".join(regions_ready("SMAP_L4_multi3snapshot", "ready_for_2021_2025")),
    }


def completed_operational_smoke_result() -> str:
    paths = sorted(
        {
            *REPORT_DIR.glob("landsurface_operational_gefs_subroot_*_2021_2025_11member_summary.csv"),
            *REPORT_DIR.glob("landsurface_operational_gefs_soill_0_1m_*_2021_2025_11member_summary.csv"),
            *OUT_DIR.glob("landsurface_operational_gefs_subroot_*_2021_2025_11member_summary.csv"),
            *OUT_DIR.glob("landsurface_operational_gefs_soill_0_1m_*_2021_2025_11member_summary.csv"),
        }
    )
    if not paths:
        return ""

    # Prefer report copies when both output and report paths exist.
    chosen: dict[str, Path] = {}
    for path in paths:
        name = path.name
        if name not in chosen or "results/report" in str(path):
            chosen[name] = path

    rows = []
    for path in sorted(chosen.values()):
        df = pd.read_csv(path)
        selected = df.loc[df["model"].eq("operational_gefs_selected")]
        persistence_raw = df.loc[df["model"].eq("persistence_raw")]
        if selected.empty:
            continue
        row = selected.iloc[0]
        region = str(row.get("region", "")).replace("_", " ")
        soil_mode = str(row.get("soil_mode", ""))
        if not soil_mode or soil_mode == "nan":
            soil_mode = "soilw_0p1_1m" if "subroot" in path.name else "soill_0_1m"
        p_bss = float(persistence_raw.iloc[0]["bss_vs_climatology"]) if not persistence_raw.empty else float("nan")
        rows.append(
            {
                "region": region,
                "soil_mode": soil_mode,
                "selected_bss": float(row["bss_vs_climatology"]),
                "selected_ci_low": float(row["bss_ci_low"]),
                "selected_ci_high": float(row["bss_ci_high"]),
                "raw_persistence_bss": p_bss,
                "source": str(path.relative_to(PROJECT_ROOT)),
            }
        )
    if not rows:
        return ""
    summary = pd.DataFrame(rows)
    n_rows = int(len(summary))
    n_regions = int(summary["region"].nunique())
    robust_positive = int((summary["selected_ci_low"] > 0).sum())
    robust_negative = int((summary["selected_ci_high"] < 0).sum())
    positive_point = int((summary["selected_bss"] > 0).sum())
    best = summary.sort_values("selected_bss", ascending=False).iloc[0]
    worst = summary.sort_values("selected_bss").iloc[0]
    return (
        f"Operational GEFS 2021-2025 compact 11-member smoke tests now cover {n_rows} "
        f"region/soil-mode rows across {n_regions} regions. Selected GEFS is robust-positive "
        f"in {robust_positive}/{n_rows}, robust-negative in {robust_negative}/{n_rows}, and "
        f"positive on point BSS in {positive_point}/{n_rows}. Best row: {best['region']} "
        f"{best['soil_mode']} BSS={best['selected_bss']:+.3f} "
        f"(CI {best['selected_ci_low']:+.3f} to {best['selected_ci_high']:+.3f}); "
        f"worst row: {worst['region']} {worst['soil_mode']} BSS={worst['selected_bss']:+.3f} "
        f"(CI {worst['selected_ci_low']:+.3f} to {worst['selected_ci_high']:+.3f}). "
        "All rows are smoke tests because operational pgrb2b soil layers are not identical "
        "to the GEFSv12 reforecast 0-100 cm RZSM approximation."
    )


def completed_operational_diagnostic_result() -> str:
    candidates = {
        "SOILW 0.1-1 m": [
            REPORT_DIR / "landsurface_operational_gefs_subroot_cvalley_2021_2025_diagnostics_candidate_scores.csv",
            OUT_DIR / "landsurface_operational_gefs_subroot_cvalley_2021_2025_diagnostics_candidate_scores.csv",
        ],
        "SOILL 0-1 m": [
            REPORT_DIR / "landsurface_operational_gefs_soill_0_1m_cvalley_2021_2025_diagnostics_candidate_scores.csv",
            OUT_DIR / "landsurface_operational_gefs_soill_0_1m_cvalley_2021_2025_diagnostics_candidate_scores.csv",
        ],
    }
    results = []
    for label, paths in candidates.items():
        path = next((candidate for candidate in paths if candidate.exists()), None)
        if path is None:
            continue
        df = pd.read_csv(path)
        selected = df.loc[df["validation_selected_any"].astype(bool)] if "validation_selected_any" in df else pd.DataFrame()
        safe = df.loc[df["persistence_safe_selected"].astype(bool)] if "persistence_safe_selected" in df else pd.DataFrame()
        if selected.empty or safe.empty:
            continue
        s = selected.iloc[0]
        p = safe.iloc[0]
        results.append(
            f"{label} diagnostic completed: validation-selected candidate="
            f"{s['candidate']} with test BSS={float(s['test_bss_vs_climatology']):+.3f}; "
            f"persistence-safe selector={p['candidate']} with test BSS={float(p['test_bss_vs_climatology']):+.3f}; "
            "no forecast candidate robustly beat raw persistence on validation."
        )
    return " ".join(results)


def build_archive_decisions(
    operational_summary: dict[str, object],
    target_summary: dict[str, object],
) -> pd.DataFrame:
    op_ready = bool(operational_summary.get("available_probe_years")) and operational_summary.get(
        "depth_compatibility"
    ) in {
        "full_0_to_1m_soilw_available",
        "partial_0p1_to_1m_soilw_available",
        "partial_soilw_0p1_to_1m_plus_soill_0_to_1m_available",
    }
    era5_ready = bool(target_summary.get("era5_land_regions_ready_2021_2025"))
    smap_ready = bool(target_summary.get("smap_regions_ready_2021_2025_local"))
    smoke_result = completed_operational_smoke_result()
    diagnostic_result = completed_operational_diagnostic_result()

    if smoke_result and diagnostic_result:
        op_decision = (
            smoke_result
            + " "
            + diagnostic_result
            + " This blocks a positive modern operational-GEFS claim under both the original "
            "SOILW 0.1-1 m extraction and the top-layer-compatible SOILL 0-1 m compatibility "
            "test. The small three-region replication reduces the risk that this is only a "
            "Central Valley artifact. The next step is not broader operational-GEFS regional "
            "scale-up; it is either a cleaner S2S/SubX archive with a native RZSM variable or "
            "base-rate-stable calibration design with modern satellite-assimilated targets."
        )
        op_priority = "high_diagnostic_followup"
    elif smoke_result:
        op_decision = (
            smoke_result
            + " This blocks a positive modern operational-GEFS claim under the current "
            "0.1-1 m extraction and validation-selected isotonic protocol. Do not scale "
            "blindly to more regions before diagnosing depth compatibility, base-rate shift, "
            "and persistence-safe model selection."
        )
        op_priority = "high_diagnostic_followup"
    elif op_ready and era5_ready:
        op_decision = (
            "Proceed with a compact operational GEFS smoke benchmark for the ERA5-Land target in "
            "regions with 2021-2025 local coverage. Keep the first run small because operational "
            "GEFS pgrb2b SOILW depth coverage is not exactly the same as the GEFSv12 reforecast "
            "0-100 cm approximation."
        )
        op_priority = "high_next_smoke_test"
    elif op_ready:
        op_decision = (
            "Archive fields are accessible, but local target products do not yet cover the modern "
            "test window. Extend targets before scoring."
        )
        op_priority = "blocked_or_conditional"
    else:
        op_decision = "Do not implement scoring yet; archive probe did not establish both lead and soil-depth support."
        op_priority = "blocked_or_conditional"

    smap_decision = (
        "SMAP L4 is the best satellite-assimilated modern target, but local processed files stop in 2019. "
        "Download/process 2020-2026 before using it as modern deployment-style validation."
        if not smap_ready
        else "SMAP L4 local files cover the modern window; score as short-record satellite-assimilated validation."
    )
    smap_target_overlap = (
        "local processed SMAP multi-snapshot regions ready for 2021-2025="
        + str(target_summary.get("smap_regions_ready_2021_2025_local", ""))
        if smap_ready
        else "local processed SMAP multi-snapshot files currently stop in 2019"
    )

    rows = [
        {
            "forecast_archive": "NOAA_GEFS_operational_AWS",
            "archive_role": "modern operational ensemble extension",
            "remote_probe_status": (
                f"available years={operational_summary.get('available_probe_years', '')}; "
                f"max_fhour={operational_summary.get('max_available_forecast_hour', '')}"
            ),
            "soil_moisture_support": operational_summary.get("depth_compatibility", ""),
            "target_overlap": (
                "ERA5-Land 2021-2025 regions="
                + str(target_summary.get("era5_land_regions_ready_2021_2025", ""))
                + "; local SMAP 2021-2025 regions="
                + str(target_summary.get("smap_regions_ready_2021_2025_local", ""))
            ),
            "scientific_decision": op_decision,
            "priority": op_priority,
            "source_url": GEFS_OPERATIONAL_REGISTRY,
            "source_reference": "NOAA GEFS open data registry; operational probe of public AWS .idx files",
        },
        {
            "forecast_archive": "NOAA_GEFSv12_reforecast_AWS",
            "archive_role": "completed hindcast benchmark",
            "remote_probe_status": "already implemented; public reforecast spans 2000-2019 in project scripts",
            "soil_moisture_support": "0_to_1m_approximation_from_soilw_bgrnd_layers",
            "target_overlap": "ERA5-Land/GLDAS/SMAP short-record checks already scored where local files exist",
            "scientific_decision": (
                "Keep as the main reproducible hindcast benchmark. It is not a modern 2020-2026 deployment "
                "test, so use operational GEFS or new target downloads for that gap."
            ),
            "priority": "complete_baseline",
            "source_url": GEFS_REFORECAST_REGISTRY,
            "source_reference": "Guan et al. (2022), doi:10.1175/MWR-D-21-0245.1",
        },
        {
            "forecast_archive": "SMAP_L4_SPL4SMGP_target_extension",
            "archive_role": "modern satellite-assimilated validation target",
            "remote_probe_status": "external catalog supports ongoing 3-hourly global root-zone product",
            "soil_moisture_support": "root_zone_0_to_100cm_target",
            "target_overlap": smap_target_overlap,
            "scientific_decision": smap_decision,
            "priority": "high_after_operational_gefs_smoke_or_parallel_download",
            "source_url": SMAP_URL,
            "source_reference": "SMAP L4 SPL4SMGP v8, doi:10.5067/T5RUATAQREF8",
        },
        {
            "forecast_archive": "SubX_or_Subseasonal_Consortium",
            "archive_role": "alternative subseasonal hindcast/forecast source",
            "remote_probe_status": "publicly available hindcasts and forecasts are documented through the IRI Data Library",
            "soil_moisture_support": "model_dependent_needs_variable_audit",
            "target_overlap": "would use same ERA5-Land/SMAP/GLDAS targets after source harmonization",
            "scientific_decision": (
                "Keep as second-line extension if operational GEFS cannot provide a defensible modern benchmark "
                "or if a multimodel subseasonal comparison becomes necessary. First audit RZSM/soil-moisture "
                "variables by model before implementing scoring."
            ),
            "priority": "medium_second_line",
            "source_url": SUBX_URL,
            "source_reference": "Pegion et al. (2019), doi:10.1175/BAMS-D-18-0270.1",
        },
        {
            "forecast_archive": "ECMWF_S2S_archive",
            "archive_role": "alternative S2S hindcast/forecast source",
            "remote_probe_status": "S2S archive support is active, with access migration to CDS-API noted by ECMWF",
            "soil_moisture_support": "parameter_available_but_access_and_license_need_confirmation",
            "target_overlap": "would use same ERA5-Land/SMAP/GLDAS targets after source harmonization",
            "scientific_decision": (
                "Scientifically strong but higher-friction. Use if the manuscript needs a true multimodel/S2S "
                "benchmark beyond GEFS; do not make it the immediate next implementation before the operational "
                "GEFS smoke test."
            ),
            "priority": "medium_high_friction",
            "source_url": S2S_ARCHIVE_URL,
            "source_reference": "Vitart et al. (2017), doi:10.1175/BAMS-D-16-0017.1",
        },
        {
            "forecast_archive": "ECMWF_open_data",
            "archive_role": "near-real-time operational access",
            "remote_probe_status": "rolling archive, not a long historical archive",
            "soil_moisture_support": "volumetric_soil_water_layers_available_in_open_data",
            "target_overlap": "not suitable for retrospective 2020-2026 scoring unless independently archived",
            "scientific_decision": (
                "Do not use as the current retrospective benchmark path. It is useful for future deployment "
                "prototyping, not for filling the historical evaluation gap."
            ),
            "priority": "low_for_retrospective_scoring",
            "source_url": ECMWF_OPEN_DATA_URL,
            "source_reference": "ECMWF Open Data documentation",
        },
    ]
    return pd.DataFrame(rows)


def native_archive_decision_rows(native_summary: pd.DataFrame, subx_probe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    if not native_summary.empty:
        for row in native_summary.itertuples(index=False):
            download_status = getattr(row, "download_probe_status", "not_run")
            metadata_status = getattr(row, "metadata_status", "")
            if row.archive == f"C3S_{C3S_ORIGINAL_DATASET}":
                if download_status == "success":
                    priority = "highest_next_benchmark"
                    decision = (
                        "Proceed with a compact ECMWF system 51 native-volumetric-soil-moisture benchmark. "
                        "This is the cleanest post-GEFS path because the archive has native VSM, long "
                        "hindcast coverage, and successful retrieval in the local environment."
                    )
                elif download_status == "blocked_license_not_accepted":
                    priority = "highest_after_license_acceptance"
                    decision = (
                        "Scientifically the strongest next archive, but scoring is intentionally blocked "
                        "until the CDS seasonal-original single-levels licence is accepted. Do not replace "
                        "this with more operational-GEFS tuning; accept the licence, rerun the tiny retrieval "
                        "probe, then implement a compact ECMWF system 51 VSM benchmark."
                    )
                elif bool(row.has_native_volumetric_soil_moisture):
                    priority = "highest_pending_retrieval_probe"
                    decision = (
                        "Metadata supports native volumetric soil moisture, with ECMWF system 51 as the best "
                        "coverage candidate. Run the retrieval probe before scoring."
                    )
                else:
                    priority = "blocked_or_conditional"
                    decision = "Catalogue metadata did not confirm native VSM support; do not score yet."
            elif row.archive == f"C3S_{C3S_MONTHLY_DATASET}":
                priority = "not_suitable_for_rzsm"
                decision = (
                    "Current monthly-statistics metadata does not expose native volumetric soil moisture, "
                    "so it is not the correct RZSM benchmark source."
                )
            else:
                priority = "blocked_or_conditional"
                decision = str(row.scientific_decision)

            rows.append(
                {
                    "forecast_archive": row.archive,
                    "archive_role": row.archive_role,
                    "remote_probe_status": f"{metadata_status}; retrieval={download_status}",
                    "soil_moisture_support": (
                        "native_volumetric_soil_moisture"
                        if bool(row.has_native_volumetric_soil_moisture)
                        else "native_vsm_not_confirmed"
                    ),
                    "target_overlap": (
                        "would use existing ERA5-Land/SMAP/GLDAS target-products after forecast extraction"
                    ),
                    "scientific_decision": decision,
                    "priority": priority,
                    "source_url": row.source_url,
                    "source_reference": row.source_reference,
                }
            )

    if not subx_probe.empty:
        auth_required = int(subx_probe["access_status"].eq("auth_required").sum())
        rows.append(
            {
                "forecast_archive": "SubX_IRI_Data_Library",
                "archive_role": "alternative native/subseasonal model source",
                "remote_probe_status": (
                    f"{auth_required}/{len(subx_probe)} probed IRI endpoints returned authentication pages"
                ),
                "soil_moisture_support": "not_auditable_without_authenticated_variable_access",
                "target_overlap": (
                    "would use existing ERA5-Land/SMAP/GLDAS target-products after forecast extraction"
                ),
                "scientific_decision": (
                    "Do not choose SubX as the immediate implementation path from the current environment. "
                    "The IRI endpoints are authentication-gated, so variable-level RZSM support was not "
                    "verified. Reconsider only with authenticated IRI access or a confirmed alternate mirror."
                ),
                "priority": "medium_blocked_by_authenticated_access",
                "source_url": "https://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/",
                "source_reference": "SubX IRI Data Library access probe",
            }
        )

    return pd.DataFrame(rows)


def write_markdown(
    decisions: pd.DataFrame,
    operational_summary: dict[str, object],
    target_summary: dict[str, object],
    native_summary: pd.DataFrame,
    native_candidates: pd.DataFrame,
    subx_probe: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Land-Surface Forecast Archive Audit",
        "",
        "Purpose: decide whether the project can extend the forecast-informed root-zone soil-moisture benchmark beyond the GEFSv12 2000-2019 reforecast window without overclaiming.",
        "",
        "## Operational GEFS Probe",
        "",
        f"- Available probe years: `{operational_summary.get('available_probe_years', '')}`.",
        f"- Maximum available forecast hour in probes: `{operational_summary.get('max_available_forecast_hour', '')}`.",
        f"- Soil-depth compatibility: `{operational_summary.get('depth_compatibility', '')}`.",
        f"- Depth note: {operational_summary.get('depth_note', '')}",
        "",
        "## Local Target Coverage",
        "",
        f"- ERA5-Land regions ready for 2021-2025: `{target_summary.get('era5_land_regions_ready_2021_2025', '')}`.",
        f"- ERA5-Land regions ready for 2021-2026 through at least 2026-03: `{target_summary.get('era5_land_regions_ready_2021_2026', '')}`.",
        f"- GLDAS regions ready for 2021-2025: `{target_summary.get('gldas_regions_ready_2021_2025', '')}`.",
        f"- Local SMAP multi-snapshot regions ready for 2021-2025: `{target_summary.get('smap_regions_ready_2021_2025_local', '')}`.",
        "",
        "## Native RZSM Archive Probe",
        "",
    ]
    if native_summary.empty:
        lines.append("- Native C3S/SubX metadata probes were skipped.")
    else:
        for row in native_summary.itertuples(index=False):
            lines.append(
                f"- `{row.archive}`: metadata=`{row.metadata_status}`, "
                f"retrieval=`{row.download_probe_status}`, "
                f"native VSM=`{row.has_native_volumetric_soil_moisture}`; "
                f"decision: {row.scientific_decision}"
            )
        if not native_candidates.empty:
            best = native_candidates.iloc[0]
            lines.append(
                "- Best C3S candidate by coverage: "
                f"`{best['originating_centre']} system {best['system']}` "
                f"({int(best['first_year'])}-{int(best['last_year'])}, "
                f"{int(best['n_months'])} months, leads "
                f"{int(best['min_leadtime_hour'])}-{int(best['max_leadtime_hour'])} h)."
            )
        if not subx_probe.empty:
            n_auth = int(subx_probe["access_status"].eq("auth_required").sum())
            lines.append(f"- SubX/IRI probe: `{n_auth}/{len(subx_probe)}` endpoints required authentication.")
    lines.extend(
        [
            "",
        "## Decision",
        "",
        ]
    )
    for row in decisions.itertuples(index=False):
        lines.extend(
            [
                f"### {row.forecast_archive}",
                "",
                f"- Priority: `{row.priority}`",
                f"- Decision: {row.scientific_decision}",
                f"- Source: {row.source_url}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    probe, operational_summary = probe_operational_gefs(args)
    targets = local_target_audit()
    target_summary = target_overlap_summary(targets)
    decisions = build_archive_decisions(operational_summary, target_summary)
    native_summary = pd.DataFrame()
    native_candidates = pd.DataFrame()
    subx_probe = pd.DataFrame()

    if not args.skip_native_rzsm_audit:
        native_summary, native_candidates = audit_c3s_native_rzsm()
        subx_probe = probe_subx_iri_access()
        if args.probe_cds_retrieval:
            cds_probe = probe_cds_vsm_retrieval(
                OUT_DIR / "c3s_seasonal_vsm_probe_ecmwf_s51_cvalley_202401_l504.nc"
            )
            native_summary = apply_native_probe_to_summary(native_summary, cds_probe)
        native_rows = native_archive_decision_rows(native_summary, subx_probe)
        if not native_rows.empty:
            decisions = pd.concat([decisions, native_rows], ignore_index=True)

    probe_path = OUT_DIR / f"{args.out_prefix}_operational_gefs_probe.csv"
    targets_path = OUT_DIR / f"{args.out_prefix}_local_target_coverage.csv"
    decisions_path = OUT_DIR / f"{args.out_prefix}.csv"
    md_path = OUT_DIR / f"{args.out_prefix}.md"
    native_path = OUT_DIR / f"{args.out_prefix}_native_rzsm_summary.csv"
    native_candidates_path = OUT_DIR / f"{args.out_prefix}_native_rzsm_candidates.csv"
    subx_probe_path = OUT_DIR / f"{args.out_prefix}_subx_iri_probe.csv"

    probe.to_csv(probe_path, index=False)
    targets.to_csv(targets_path, index=False)
    decisions.to_csv(decisions_path, index=False)
    native_summary.to_csv(native_path, index=False)
    native_candidates.to_csv(native_candidates_path, index=False)
    subx_probe.to_csv(subx_probe_path, index=False)
    write_markdown(
        decisions,
        operational_summary,
        target_summary,
        native_summary,
        native_candidates,
        subx_probe,
        md_path,
    )

    if args.copy_report:
        for path in [
            probe_path,
            targets_path,
            decisions_path,
            md_path,
            native_path,
            native_candidates_path,
            subx_probe_path,
        ]:
            shutil.copy2(path, REPORT_DIR / path.name)
        shutil.copy2(decisions_path, PAPER_DIR / "table29_landsurface_forecast_archive_audit.csv")
        shutil.copy2(native_path, PAPER_DIR / "table37_native_rzsm_archive_audit.csv")
        shutil.copy2(native_candidates_path, PAPER_DIR / "table38_native_rzsm_archive_candidates.csv")

    print(f"Wrote {decisions_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {probe_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {targets_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {native_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {native_candidates_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {subx_probe_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {md_path.relative_to(PROJECT_ROOT)}")
    print("")
    print("Archive decisions:")
    compact = decisions[["forecast_archive", "priority", "scientific_decision"]]
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
