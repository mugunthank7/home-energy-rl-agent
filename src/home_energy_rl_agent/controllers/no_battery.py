"""No-battery baseline: always idle. This is the cost reference for every savings number."""
from home_energy_rl_agent.controllers.base import ActionMap


class NoBattery:
    name = "no_battery"

    def __init__(self, actions: ActionMap):
        self.actions = actions

    def reset(self) -> None:
        pass

    def act(self, obs) -> int:
        return self.actions.idle
