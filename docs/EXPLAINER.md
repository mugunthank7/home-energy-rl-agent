# Module explainers

Plain-English notes so each module can be explained on video. One entry per module.

## data.py (Phase 1)
- **What it does:** turns the raw UCI file (about 2 million minute-level rows) into a clean hourly demand series in kW. This is the REAL data.
- **Parsing:** the file uses `;` separators, dd/mm/yyyy dates, and `?` for missing values. We combine Date and Time into one timestamp and turn `?` into NaN.
- **Gap handling:** missing stretches of 3 hours or less are forward-filled, because a short sensor dropout probably looks like the minutes before it. Longer gaps are not guessed.
- **Hourly resample:** the mean kW over an hour equals the kWh used in that hour. An hour with any missing minute stays missing, so partial hours never look complete.
- **Dropping days:** any calendar day that does not have all 24 hours is dropped, and the counts are logged in `data/processed/cleaning_log.json`.
- **Splits:** by calendar year and never shuffled: train 2007-08, validation 2009, test 2010. Shuffling would leak the future into training.
- **Held-out weeks:** a winter week (Jan 18-24, 2010) and a summer week (Jul 5-11, 2010) are kept for the generalization test. Jan 12-14 have long gaps, and August is avoided because the household was away.
- **Episode starts:** because dropped days leave holes, `valid_episode_starts` only allows 7-day windows that are fully contiguous.
- **Why it matters:** the RL agent trains on the train split only. Everything else (battery, tariff) is simulated on top of this real demand.
