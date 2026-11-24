# Methodology

## Real vs simulated
- REAL: household demand (UCI dataset id 235).
- SIMULATED: battery model, time-of-use tariff, demand charge, any PV.

## Environment (Phase 2)
- Step = 1 hour; episode = 7 days drawn from contiguous windows of the chosen split.
- Battery power is on the AC side; round-trip efficiency is split as sqrt per direction.
- Violation = requested charge/discharge beyond SOC bounds or the power limit (clipped and penalized).
- Discharge is capped at current demand (no export). The cut amount is logged as `curtailed_kw` and is not penalized.
- Reward = -(price x grid import) - w_peak x (increase in episode peak) - w_viol x violation - w_cycle x throughput.
- The demand charge (price per kW of the episode's peak import) is added to the reported total cost, while the reward only uses the peak-increase term as shaping.

(Further details are filled in as each phase lands.)
