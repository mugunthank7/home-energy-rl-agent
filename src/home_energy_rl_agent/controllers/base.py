"""Shared controller interface and helpers for reading observations and actions.

Every controller implements `act(obs) -> action` (an integer index) and `reset()`.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]


def load_controller_config(path: str | Path = ROOT / "configs" / "controllers.yaml") -> dict:
    """Read baseline controller settings from YAML."""
    return yaml.safe_load(Path(path).read_text())


class Controller(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(self, obs: np.ndarray) -> int: ...


@dataclass(frozen=True)
class ActionMap:
    """Which action index means what, derived from the env's power fractions."""

    discharge_high: int
    discharge_low: int
    idle: int
    charge_low: int
    charge_high: int

    @classmethod
    def from_fractions(cls, fractions) -> "ActionMap":
        f = np.asarray(fractions, dtype=float)
        neg, pos = np.flatnonzero(f < 0), np.flatnonzero(f > 0)
        return cls(
            discharge_high=int(neg[np.argmin(f[neg])]),
            discharge_low=int(neg[np.argmax(f[neg])]),
            idle=int(np.flatnonzero(f == 0)[0]),
            charge_low=int(pos[np.argmin(f[pos])]),
            charge_high=int(pos[np.argmax(f[pos])]),
        )


class ObsReader:
    """Pulls named values out of the observation vector."""

    def __init__(self, obs_names: list[str]):
        self.i = {name: k for k, name in enumerate(obs_names)}

    def soc(self, obs) -> float:
        return float(obs[self.i["soc"]])

    def period(self, obs) -> int:
        return int(obs[self.i["tariff_period"]])

    def demand(self, obs) -> float:
        return float(obs[self.i["demand_kw"]])
