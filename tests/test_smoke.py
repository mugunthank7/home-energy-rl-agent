"""Smoke test: the package imports and the configs parse."""
from pathlib import Path

import yaml

import home_energy_rl_agent

ROOT = Path(__file__).resolve().parents[1]


def test_package_imports():
    assert home_energy_rl_agent.__doc__


def test_configs_parse():
    env = yaml.safe_load((ROOT / "configs" / "env.yaml").read_text())
    tariff = yaml.safe_load((ROOT / "configs" / "tariff.yaml").read_text())
    assert env["battery"]["capacity_kwh"] == 10.0
    assert tariff["prices_per_kwh"]["on_peak"] > tariff["prices_per_kwh"]["off_peak"]
