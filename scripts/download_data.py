"""Download the UCI Individual Household Electric Power Consumption dataset (id 235).

Run manually: `uv run python scripts/download_data.py`. Raw data is gitignored.
"""
import io
import zipfile
from pathlib import Path

import requests

# --- Where the data comes from and where it goes ---
URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / "household_power_consumption.txt"
    if target.exists():
        print(f"Already downloaded: {target}")
        return
    # --- Download the zip into memory and extract the text file ---
    print(f"Downloading {URL}")
    resp = requests.get(URL, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(RAW_DIR)
    print(f"Saved to {RAW_DIR}")


if __name__ == "__main__":
    main()
