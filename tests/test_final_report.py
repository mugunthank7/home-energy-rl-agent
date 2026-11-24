"""Tests for the final tables, bar charts, day plots, policy heatmaps and parameter record."""
import numpy as np
import pandas as pd
import pytest
import yaml

from home_energy_rl_agent import final_report as fr
from home_energy_rl_agent.controllers.base import ActionMap
from home_energy_rl_agent.controllers.qlearning import Discretizer, QLearningAgent
from home_energy_rl_agent.env import EnergyEnv, load_env_config
from home_energy_rl_agent.final_eval import season_starts
from home_energy_rl_agent.tariff import load_tariff_config

ROOT = fr.data.ROOT
SPLITS = ["validation", "test", "test_winter_week", "test_summer_week",
          "test_winter_season", "test_summer_season"]


def fake_record(split, controller, seed, savings, n=10):
    summary = {k: 1.0 for k, _ in fr.METRICS} | {"savings_total_pct": savings,
                                                 "savings_total_adj_pct": savings - 0.5}
    per_ep = {c: [10.0 + i * 0.1 for i in range(n)] for c in
              ["reward", "energy_cost", "total_cost", "peak_kw", "violations", "equivalent_full_cycles",
               "base_energy_cost", "base_total_cost", "base_peak_kw", "final_soc", "total_cost_adj"]}
    per_ep["base_total_cost"] = [20.0] * n
    return {"split": split, "controller": controller, "seed": seed, "summary": summary, "per_episode": per_ep,
            "train_time_s": 9.0 if seed is not None else None}


@pytest.fixture
def rec():
    records = []
    for sp in SPLITS:
        records += [fake_record(sp, "no_battery", None, 0.0), fake_record(sp, "rule_based", None, 1.0),
                    fake_record(sp, "greedy_threshold", None, 9.0)]
        for seed, sav in [(1, 2.0), (2, 4.0)]:
            records += [fake_record(sp, f"q_{n}", seed, sav) for n in ("baseline", "tuned", "selected")]
        records += [fake_record(sp, "oracle", None, 40.0), fake_record(sp, "oracle_adj", None, 40.0)]
    stats = {sp: {"hours": 100, "mean_kw": 1.2, "p95_kw": 3.0, "max_kw": 6.0, "mean_weekly_peak_kw": 4.0}
             for sp in ["train", "validation", "test"]}
    return {"seeds": [1, 2], "records": records, "split_stats": stats}


def test_metrics_table_has_every_controller_and_mean_std_for_q(rec):
    df = fr.metrics_table(rec, "test")
    assert list(df["controller"]) == [fr.LABEL[c] for c in fr.ORDER]       # oracle is the last row
    q_row = df[df["controller"] == fr.LABEL["q_selected"]].iloc[0]
    assert q_row["total savings %"] == "3.00 ± 1.00"           # mean and std over the two seeds
    assert df[df["controller"] == fr.LABEL["rule_based"]].iloc[0]["total savings %"] == "1.00"
    assert q_row["% of oracle savings"] == "7.5 ± 2.5"             # (2/40, 4/40) as percentages
    shares = [100 * 1.5 / 39.5, 100 * 3.5 / 39.5]                 # adjusted savings of the two seeds
    assert q_row["% of oracle (terminal-adjusted)"] == f"{np.mean(shares):.1f} ± {np.std(shares):.1f}"
    assert df[df["controller"] == fr.LABEL["greedy_threshold"]].iloc[0]["% of oracle savings"] == "22.5"
    assert df[df["controller"] == fr.LABEL["oracle"]].iloc[0]["% of oracle savings"] == "100.0"
    assert q_row["total savings, terminal-adjusted %"] == "2.50 ± 1.00"
    assert df[df["controller"] == fr.LABEL["no_battery"]].iloc[0]["train time (s)"].startswith("0")


def test_generalization_table_reports_val_to_test_change(rec):
    for r in rec["records"]:                                    # make test 1 point worse than validation
        if r["split"] == "test" and r["controller"] == "greedy_threshold":
            r["summary"]["savings_total_pct"] = 8.0
    gen = fr.generalization_table(rec)
    row = gen[gen["controller"] == fr.LABEL["greedy_threshold"]].iloc[0]
    assert row["change val to test (pts)"] == "-1.0"
    assert gen[gen["controller"] == fr.LABEL["no_battery"]].iloc[0]["change val to test (pts)"] == "n/a"


def test_markdown_has_header_and_one_row_per_controller(rec):
    md = fr.df_to_markdown(fr.metrics_table(rec, "test"), "Test", "note")
    assert md.startswith("### Test") and md.count("\n| ") == len(fr.ORDER) + 1


def test_split_stats_table(rec):
    assert list(fr.split_stats_table(rec)["split"]) == ["train", "validation", "test"]


def test_controller_bars_written(rec, tmp_path):
    fr.plot_controller_bars(rec, "test", tmp_path / "bars.png")
    assert (tmp_path / "bars.png").stat().st_size > 1000


def test_plot_days_written(tmp_path):
    def day(price):
        return [{"demand": 1.0, "grid_import": 1.2, "soc": 0.5, "action": 4, "price": price}] * 24

    days = {"winter": {"rule_based": day(0.3), "q_selected": day(0.3)},
            "summer": {"rule_based": day(0.15), "q_selected": day(0.15)}}
    fr.plot_days(days, tmp_path / "days.png")
    assert (tmp_path / "days.png").exists()


def make_agent(trained: bool):
    idx = pd.date_range("2008-03-03", periods=60 * 24, freq="h")
    hours = idx.hour
    demand = pd.Series(0.5 + 1.2 * np.exp(-((hours - 19) ** 2) / 8), index=idx)
    env = EnergyEnv(demand, load_env_config(), load_tariff_config(), seed=0)
    cfg = yaml.safe_load((ROOT / "configs" / "experiments" / "selected.yaml").read_text())
    disc = Discretizer.fit(env.obs_names, demand, env.battery.p, cfg["discretization"])
    agent = QLearningAgent(disc, 5, ActionMap.from_fractions(env.fractions).idle, cfg["qlearning"])
    if trained:
        agent.train(env, 30, seed=0)
    return agent


def test_policy_grid_shape_and_unvisited_marker():
    untrained = make_agent(False)
    assert (fr.policy_grid(untrained, 0) == -1).all()           # nothing visited yet
    trained = make_agent(True)
    grid = fr.policy_grid(trained, 1)
    assert grid.shape == (trained.disc.soc_bins, trained.disc.hour_buckets)
    assert grid.min() >= -1 and grid.max() <= 4
    assert (grid >= 0).any()


def test_policy_heatmap_written(tmp_path):
    fr.plot_policy_heatmap(make_agent(True), "t", tmp_path / "heat.png")
    assert (tmp_path / "heat.png").exists()


def test_season_starts_only_midnight_starts_in_the_given_months():
    idx = pd.date_range("2010-01-01", "2010-04-30 23:00", freq="h")
    env = EnergyEnv(pd.Series(1.0, index=idx), load_env_config(), load_tariff_config(), seed=0)
    starts = season_starts(env, [1, 2])
    times = env.series.index[starts]
    assert len(starts) > 0
    assert set(times.hour) == {0} and set(times.month) <= {1, 2}


def test_parameter_record_lists_configs_and_sweeps(tmp_path):
    fr.parameter_record(tmp_path / "params.md")
    text = (tmp_path / "params.md").read_text()
    assert "capacity_kwh" in text and "sweep_alpha" in text and "| selected |" in text


def test_table_without_oracle_says_how_to_get_it(rec):
    rec = {**rec, "records": [r for r in rec["records"] if not r["controller"].startswith("oracle")]}
    df = fr.metrics_table(rec, "test")
    assert "run_oracle" in df.iloc[0]["% of oracle savings"]
    assert list(df["controller"]) == [fr.LABEL[c] for c in fr.ORDER if c != "oracle"]


def test_oracle_share_table_covers_every_split(rec):
    share = fr.oracle_share_table(rec)
    assert list(share.columns)[1:] == [fr.SPLIT_TITLE[s] for s in SPLITS]
    assert share[share["controller"] == fr.LABEL["greedy_threshold"]].iloc[0, 1] == "22.5"
