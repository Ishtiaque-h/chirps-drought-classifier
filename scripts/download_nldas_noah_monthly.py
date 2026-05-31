#!/usr/bin/env python
"""Download NLDAS Noah monthly soil-moisture files from NASA GES DISC.

The NLDAS monthly catalog is browsable without credentials, but NetCDF file
downloads require Earthdata/GES DISC authorization. This script supports either
an existing ``~/.netrc`` entry for ``urs.earthdata.nasa.gov`` or explicit
``--username`` plus a password stored in an environment variable.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import netrc
import os
import shutil
import subprocess

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nldas_noah_monthly"
BASE_URL = "https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_NOAH0125_M.2.0"
USER_AGENT = "chirps-drought-classifier/nldas-independent-target"


class DownloadError(RuntimeError):
    pass


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1991-01", help="First monthly file, YYYY-MM.")
    parser.add_argument("--end", default="2019-12", help="Last monthly file, YYYY-MM.")
    parser.add_argument("--out-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--username", default=None, help="Earthdata username. If omitted, requests may use ~/.netrc.")
    parser.add_argument(
        "--password-env",
        default="EARTHDATA_PASSWORD",
        help="Environment variable containing the Earthdata password when --username is used.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--backend",
        choices=["curl", "requests"],
        default="curl",
        help="Use curl+netrc by default because NASA Earthdata redirects are more reliable through curl.",
    )
    return parser.parse_args()


def month_starts(start: str, end: str) -> list[pd.Timestamp]:
    months = pd.date_range(
        pd.Timestamp(start).to_period("M").to_timestamp(),
        pd.Timestamp(end).to_period("M").to_timestamp(),
        freq="MS",
    )
    return [pd.Timestamp(month) for month in months]


def nldas_filename(month: pd.Timestamp) -> str:
    return f"NLDAS_NOAH0125_M.A{month:%Y%m}.020.nc"


def nldas_url(month: pd.Timestamp) -> str:
    return f"{BASE_URL}/{month:%Y}/{nldas_filename(month)}"


def destination(out_dir: Path, month: pd.Timestamp) -> Path:
    return out_dir / f"{month:%Y}" / nldas_filename(month)


def build_session(args: Namespace) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    if args.username:
        password = os.environ.get(args.password_env)
        if not password:
            raise DownloadError(
                f"--username was provided, but ${args.password_env} is not set. "
                "Set the password in the environment or configure ~/.netrc."
            )
        session.auth = (args.username, password)
    elif not has_netrc_credentials():
        raise DownloadError(
            "No Earthdata credentials found. Configure ~/.netrc for urs.earthdata.nasa.gov "
            "or pass --username with the password in $EARTHDATA_PASSWORD. Also approve the NASA GES DISC app."
        )
    return session


def has_netrc_credentials() -> bool:
    try:
        auths = netrc.netrc()
    except (FileNotFoundError, netrc.NetrcParseError):
        return False
    for host in ["urs.earthdata.nasa.gov", "hydro1.gesdisc.eosdis.nasa.gov"]:
        try:
            if auths.authenticators(host):
                return True
        except (FileNotFoundError, netrc.NetrcParseError):
            return False
    return False


def looks_like_netcdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as handle:
        magic = handle.read(8)
    return magic.startswith(b"CDF") or magic.startswith(b"\x89HDF")


def validate_download(path: Path) -> None:
    if looks_like_netcdf(path):
        return
    try:
        head = path.read_bytes()[:256]
    except FileNotFoundError as exc:
        raise DownloadError(f"Download did not create {path}") from exc
    text = head.decode("utf-8", errors="ignore")
    if "Access denied" in text or "Earthdata" in text or "Basic" in text:
        raise DownloadError(
            "NLDAS file download was denied. Configure Earthdata credentials and approve the NASA GES DISC app."
        )
    raise DownloadError(f"Downloaded file is not a NetCDF file: {path} ({path.stat().st_size} bytes)")


def download_one(args: Namespace, session: requests.Session | None, month: pd.Timestamp, out_dir: Path, overwrite: bool) -> str:
    path = destination(out_dir, month)
    if path.exists() and looks_like_netcdf(path) and not overwrite:
        return "cached"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    url = nldas_url(month)
    if args.backend == "curl":
        download_with_curl(url, tmp, path.parent)
        validate_download(tmp)
        shutil.move(tmp, path)
        return "downloaded"
    if session is None:
        raise DownloadError("Internal error: requests backend selected without a session.")
    download_with_requests(session, url, tmp)
    validate_download(tmp)
    shutil.move(tmp, path)
    return "downloaded"


def download_with_requests(session: requests.Session, url: str, tmp: Path) -> None:
    try:
        with session.get(url, stream=True, timeout=120, allow_redirects=True) as response:
            if response.status_code in {401, 403}:
                raise DownloadError(f"Access denied for {url}; configure Earthdata credentials.")
            if response.status_code >= 400:
                raise DownloadError(f"HTTP {response.status_code} while downloading {url}")
            with tmp.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        validate_download(tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def download_with_curl(url: str, tmp: Path, cookie_dir: Path) -> None:
    if not has_netrc_credentials():
        raise DownloadError(
            "No Earthdata credentials found. Configure ~/.netrc for urs.earthdata.nasa.gov "
            "and approve the NASA GES DISC app."
        )
    cookies = cookie_dir / ".earthdata_cookies"
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--netrc-file",
        str(Path.home() / ".netrc"),
        "-c",
        str(cookies),
        "-b",
        str(cookies),
        "--connect-timeout",
        "30",
        "--max-time",
        "300",
        "--retry",
        "3",
        "--retry-delay",
        "2",
        "-o",
        str(tmp),
        url,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"curl failed for {url} with exit code {exc.returncode}") from exc


def main() -> None:
    args = parse_args()
    months = month_starts(args.start, args.end)
    if args.max_files is not None:
        months = months[: args.max_files]

    print(f"NLDAS_NOAH0125_M monthly files: {len(months)}")
    print(f"Output directory: {args.out_dir}")
    print(
        f"Credential mode: {'explicit username' if args.username else 'netrc'}; "
        f"backend={args.backend}",
        flush=True,
    )
    if args.dry_run:
        for month in months[:12]:
            print(f"{month:%Y-%m}: {nldas_url(month)} -> {destination(args.out_dir, month)}")
        if len(months) > 12:
            print(f"... {len(months) - 12} additional files")
        return

    session = build_session(args) if args.backend == "requests" else None
    counts: dict[str, int] = {}
    for i, month in enumerate(months, start=1):
        status = download_one(args, session, month, args.out_dir, args.overwrite)
        counts[status] = counts.get(status, 0) + 1
        print(f"[{i:03d}/{len(months):03d}] {month:%Y-%m}: {status}", flush=True)
    print("Done:", counts)


if __name__ == "__main__":
    try:
        main()
    except DownloadError as exc:
        raise SystemExit(str(exc)) from exc
