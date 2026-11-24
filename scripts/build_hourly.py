"""Build data/processed/hourly.parquet from the raw UCI file and print the cleaning log."""
import json

from home_energy_rl_agent import data


def main() -> None:
    # --- Clean, resample, save ---
    cfg = data.load_config()
    hourly, log = data.build_hourly(cfg)
    data.save_hourly(hourly, log, cfg)
    print(json.dumps(log, indent=2))

    # --- Report split sizes and how many 7-day episodes each split supports ---
    for name, series in data.get_splits(hourly, cfg).items():
        starts = data.valid_episode_starts(series)
        print(f"{name}: {len(series)} hours, {len(starts)} valid 7-day episode starts")


if __name__ == "__main__":
    main()
