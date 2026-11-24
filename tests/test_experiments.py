"""Tests for sweep expansion, config overrides, and result aggregation (no training needed)."""
import json

import numpy as np
import pytest

from home_energy_rl_agent import experiments as ex
from home_energy_rl_agent import reporting


def test_set_path_creates_nested_keys():
    cfg = {"a": {"b": 1}}
    ex.set_path(cfg, "a.c.d", 5)
    ex.set_path(cfg, "x", 2)
    assert cfg == {"a": {"b": 1, "c": {"d": 5}}, "x": 2}


def test_deep_merge_does_not_mutate_inputs():
    base = {"r": {"w": 1, "v": 2}, "k": 3}
    out = ex.deep_merge(base, {"r": {"w": 9}})
    assert out == {"r": {"w": 9, "v": 2}, "k": 3}
    assert base["r"]["w"] == 1


def test_expand_factor_values():
    v = ex.expand_variants({"factor": "qlearning.alpha", "values": [0.01, 0.1]})
    assert [x["label"] for x in v] == ["alpha=0.01", "alpha=0.1"]
    assert v[1]["set"] == {"qlearning.alpha": 0.1}


def test_expand_grid_is_the_full_product_in_order():
    v = ex.expand_variants({"grid": {"qlearning.alpha": [1, 2], "qlearning.gamma": [7, 8, 9]}})
    assert len(v) == 6
    assert v[0]["set"] == {"qlearning.alpha": 1, "qlearning.gamma": 7}
    assert v[-1]["set"] == {"qlearning.alpha": 2, "qlearning.gamma": 9}
    assert len({x["label"] for x in v}) == 6


def test_resolve_variant_applies_overrides_to_a_copy():
    base = {"qlearning": {"alpha": 0.1}, "env_overrides": {}}
    out = ex.resolve_variant(base, {"label": "x", "set": {"qlearning.alpha": 0.5,
                                                          "env_overrides.battery.capacity_kwh": 20}})
    assert out["qlearning"]["alpha"] == 0.5
    assert out["env_overrides"]["battery"]["capacity_kwh"] == 20
    assert base["qlearning"]["alpha"] == 0.1 and base["env_overrides"] == {}


def test_run_id_is_filesystem_safe():
    assert ex.run_id("coarse (5x6x3x3)", 42) == "coarse_5x6x3x3_seed42"
    assert ex.run_id("alpha=0.01,gamma=0.9", 1) == "alpha_0_01_gamma_0_9_seed1"


def test_every_sweep_config_is_valid():
    """Guards against typos: unique labels, base exists, learning/grid overrides hit real keys."""
    sweeps = sorted(ex.EXP_DIR.glob("sweep_*.yaml"))
    assert len(sweeps) >= 7                                  # the plan asks for at least 3, we do 7+
    for path in sweeps:
        sweep = ex.load_experiment(path.stem)
        base = ex.load_experiment(sweep["base"])
        variants = ex.expand_variants(sweep)
        assert len({v["label"] for v in variants}) == len(variants), path.stem
        assert sweep["name"] == path.stem
        for v in variants:
            resolved = ex.resolve_variant(base, v)
            for dotted, value in v["set"].items():
                top, *_rest, leaf = dotted.split(".")
                if top in ("qlearning", "discretization"):
                    assert leaf in base[top], f"{path.stem}: unknown key {dotted}"
                node = resolved
                for key in dotted.split("."):
                    node = node[key]
                assert node == value


def test_build_jobs_has_one_job_per_variant_and_seed():
    jobs = ex.build_jobs("sweep_alpha", seeds=[1, 2])
    assert len(jobs) == 4 * 2
    assert {j["seed"] for j in jobs} == {1, 2}
    assert [j["order"] for j in jobs[::2]] == [0, 1, 2, 3]


def make_fake_run(order, label, seed, savings):
    summary = {m: 0.0 for m in reporting.TABLE_METRICS} | {"savings_total_pct": savings}
    n = 300
    return {"sweep": "sweep_fake", "label": label, "order": order, "seed": seed, "set": {},
            "summary": summary,
            "history": {"episode_reward": list(np.linspace(-100, -30, n)),
                        "eval": [{"episode": 150, "reward": -40.0}, {"episode": 300, "reward": -35.0}]}}


@pytest.fixture
def fake_results(tmp_path):
    d = tmp_path / "sweep_fake"
    d.mkdir()
    runs = [make_fake_run(1, "b", 1, 4.0), make_fake_run(0, "a", 1, 1.0),
            make_fake_run(0, "a", 2, 3.0), make_fake_run(1, "b", 2, 6.0)]
    for i, r in enumerate(runs):
        (d / f"run{i}.json").write_text(json.dumps(r))
    return tmp_path


def test_aggregate_orders_variants_and_takes_mean_std_over_seeds(fake_results):
    agg = reporting.aggregate(reporting.load_runs("sweep_fake", fake_results))
    assert list(agg["label"]) == ["a", "b"]                  # config order, not alphabetical by seed
    assert agg.loc[0, "savings_total_pct_mean"] == pytest.approx(2.0)
    assert agg.loc[0, "savings_total_pct_std"] == pytest.approx(1.0)
    assert list(agg["seeds"]) == [2, 2]


def test_markdown_table_has_a_row_per_variant(fake_results):
    agg = reporting.aggregate(reporting.load_runs("sweep_fake", fake_results))
    md = reporting.to_markdown(agg, "fake")
    assert "| a | 2.00 ± 1.00" in md and "| b | 5.00 ± 1.00" in md


def test_plot_sweep_writes_both_figures(fake_results, tmp_path):
    runs = reporting.load_runs("sweep_fake", fake_results)
    reporting.plot_sweep("sweep_fake", runs, reporting.aggregate(runs), fig_dir=tmp_path / "figs")
    assert (tmp_path / "figs" / "sweep_fake_convergence.png").exists()
    assert (tmp_path / "figs" / "sweep_fake_final.png").exists()
