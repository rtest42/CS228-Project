import csv
import os
import re
import threading
import urllib.request
import zipfile

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Final, TypedDict

# tqdm's default lock is a multiprocessing lock, which starts a helper
# process. On macOS, starting a process after worker threads have used the
# network crashes that helper (Python "quit unexpectedly" popups). The
# progress bars are only shared across threads, so a thread lock suffices.
tqdm.set_lock(threading.RLock())

NUM_WORKERS: Final[int] = 4
CHUNK_SIZE: Final[int] = 64 * 1024

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
IMAGE_DIR: Final[Path] = DATA_DIR / "processed"
MANIFEST_PATH: Final[Path] = DATA_DIR / "manifest.csv"

# noinspection HttpUrlsUsage
URLS: Final[list[str]] = [
    f"http://bias.csr.unibo.it/{year}/Downloads/DB{i}_B.zip"
    for year in ["fvc2000", "fvc2002", "fvc2004"]
    for i in range(1, 5)
]

FILENAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<finger>\d+)_(?P<impression>\d+)\.(?:tif|tiff)$",
    re.IGNORECASE,
)


class ManifestRow(TypedDict):
    image_path: str
    class_id: str
    dataset: str
    database: str
    finger_id: str
    impression_id: int


def download_file(url: str) -> Path:
    """Download a ZIP archive, reusing an existing valid archive."""
    parts: list[str] = url.split("/")
    year: str = parts[3]
    filename: str = parts[-1]

    output_dir: Path = RAW_DIR / year
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path: Path = output_dir / filename
    temp_path: Path = output_dir / (filename + ".part")

    # Reuse existing valid archive.
    if zip_path.exists() and zipfile.is_zipfile(zip_path):
        return zip_path

    try:
        request: urllib.request.Request = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}
        )

        with urllib.request.urlopen(request, timeout=60) as response:
            total_size: int = int(response.headers.get("Content-Length", 0))

            with open(temp_path, "wb") as f, tqdm(
                    total=total_size or None,
                    desc=f"Download {year}/{filename}",
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    position=None,
                    leave=False,
            ) as pbar:
                while True:
                    chunk: bytes = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))

        if not zipfile.is_zipfile(temp_path):
            raise ValueError("Downloaded file is not a valid ZIP archive.")

        os.replace(temp_path, zip_path)
        return zip_path

    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def extract_file(zip_path: Path) -> Path:
    """Extract a ZIP archive into its own dataset directory."""
    year: str = zip_path.parent.name
    database: str = zip_path.stem

    output_dir: Path = IMAGE_DIR / year / database
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            members: list[zipfile.ZipInfo] = archive.infolist()
            root: Path = output_dir.resolve()

            # Validate all archive paths before extracting.
            for member in members:
                target: Path = (output_dir / member.filename).resolve()

                if os.path.commonpath([str(root), str(target)]) != str(root):
                    raise ValueError(f"Unsafe ZIP path: {member.filename}")

            total_bytes: int = sum(member.file_size for member in members
                                   if not member.is_dir())

            with tqdm(
                total=total_bytes,
                desc=f"Extract {year}/{database}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                position=None,
                leave=False,
            ) as pbar:
                for member in members:
                    target = output_dir / member.filename

                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue

                    target.parent.mkdir(parents=True, exist_ok=True)

                    with archive.open(member) as src, open(target, "wb") as dst:
                        while True:
                            chunk: bytes = src.read(CHUNK_SIZE)
                            if not chunk:
                                break

                            dst.write(chunk)
                            pbar.update(len(chunk))

        return output_dir
    except Exception as err:
        raise RuntimeError(f"Could not extract {zip_path}: {err}") from err


def build_manifest() -> None:
    """Index TIFF files and write their metadata to a CSV manifest."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = sorted(
        path
        for path in IMAGE_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".tif", ".tiff"}
    )

    fields: list[str] = [
        "image_path",
        "class_id",
        "dataset",
        "database",
        "finger_id",
        "impression_id",
    ]

    count: int = 0
    unmatched: list[Path] = []

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer: csv.DictWriter[str] = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for path in image_paths:
            match: re.Match[str] | None = FILENAME_PATTERN.match(path.name)
            if match is None:
                unmatched.append(path)
                continue

            year: str = path.parents[1].name
            database: str = path.parent.name
            finger: str = match.group("finger")
            impression: int = int(match.group("impression"))

            finger_id: str = f"{year}/{database}/finger_{finger}"

            # class_id is the column the Java baseline reads; it names the
            # same finger identity as finger_id.
            row: ManifestRow = {
                "image_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "class_id": finger_id,
                "dataset": year,
                "database": database,
                "finger_id": finger_id,
                "impression_id": impression
            }

            writer.writerow(dict(row))
            count += 1

    print(f"\nManifest: {MANIFEST_PATH}")
    print(f"Indexed TIFF images: {count}")

    if unmatched:
        print(f"\nWARNING: {len(unmatched)} filenames need inspection.")
        for path in unmatched[:10]:
            print(f"\t{path}")

        print("Update FILENAME_PATTERN to match the actual filenames before training.")


def main() -> None:
    """Download archives, extract them, and build the manifest."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    print("Phase 1/3: Downloading archives")

    zip_paths: list[Path] = []

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(download_file, url): url for url in URLS}

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Downloads"
        ):
            url: str = futures[future]

            try:
                zip_paths.append(future.result())
            except Exception as err:
                print(f"\nDownload failed: {url}: {err}")

    print("\nPhase 2/3: Extracting archives")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(extract_file, path): path for path in zip_paths}

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Extractions"
        ):
            try:
                future.result()
            except Exception as err:
                print(f"\nExtraction failed: {err}")

    print("\nPhase 3/3: Building manifest")
    build_manifest()

    print("\nDataset preparation finished.")


if __name__ == "__main__":
    main()
