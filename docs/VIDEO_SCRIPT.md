# Video script (up to 8 minutes)

Spoken at a calm pace of about 130 words a minute. Everything in quotation marks is meant to be said
out loud; edit it so it sounds like you. Numbers come from `docs/REPORT_DRAFT.md`. If you re-run any
experiment, check the numbers again.

**Setup before recording:** open the repo, the README, and these files in tabs so you can switch quickly:
`reports/figures/eda_daily_profile.png`, `env_check_episode.png`, `sweep_epsilon_convergence.png`,
`sweep_grid_alpha_gamma_heatmap.png`, `controllers_test.png`, `representative_days.png`,
`policy_heatmap_selected.png`.

---

## 0:00 to 0:45 | Problem and why it matters

**Show:** `eda_daily_profile.png`.

"Homes that pay time-of-use prices and a demand charge can save money with a battery. Charge when power is
cheap, discharge when it is expensive, and avoid setting a high peak. This is the daily demand of one real
French household. It peaks at about 7 in the morning and again from 7 to 9 in the evening, when my
simulated price is highest. The question I asked is whether Q-learning can learn a battery controller that
beats simple rules, and how well it holds up on a later year and other seasons."

"One thing to be clear about from the start: the demand is real data. The battery, the prices and the
demand charge are simulated, so every saving I show is a simulation result."

## 0:45 to 2:00 | Data and environment

**Show:** the README real-vs-simulated box, then `env_check_episode.png`.

"The data is the UCI household power dataset, about two million minute readings. I turned missing values
into gaps, filled gaps up to 3 hours, averaged to hourly, and dropped 24 days that still had holes. I split
by time: 2007 and 2008 to train, 2009 to validate, 2010 to test, never shuffled."

"The environment is a Gymnasium environment. One step is an hour and one episode is a week. The battery is
10 kilowatt-hours with a 3 kilowatt limit and 90 percent round-trip efficiency. There are five actions: discharge
high or low, idle, charge low or high. This plot shows one week: demand in blue, grid import in orange,
battery level in green, and the price below. The battery charges at night and discharges in the evening
price window. The tests check that energy is conserved, the charge level stays in bounds, and results
repeat under the same seed."

## 2:00 to 3:30 | Methods

**Show:** the README controller list, or `docs/EXPLAINER.md` for the Q-learning entry.

"I compare four controllers. No battery is the reference. The rule-based controller charges at full power
overnight and discharges in the evening. The greedy controller does the same idea using price percentiles,
but it charges gently and scales discharge to current demand."

"The learner is tabular Q-learning. The state is battery level, hour, demand level and tariff period, 3,600
situations. It tries actions with epsilon-greedy exploration, and updates a table of values with a learning
rate alpha and a discount gamma."

"I changed the reward, and I want to explain why. The plan's peak penalty trained a bad agent: it was noisy
and the agent could not see the peak. I now charge only the peak the battery adds on top of the home's own
peak. That does not change the best policy, but it removes a lot of noise. I also raised the violation
penalty, because the agent was breaking battery limits on purpose when it was cheap to do so."

"I also built an upper bound, an oracle. It is a linear program that is told the whole week of demand and
prices in advance, and finds the cheapest possible battery plan under the same rules. No real controller can
do that, but it tells me how much is actually achievable."

## 3:30 to 5:30 | Results

**Show:** `sweep_epsilon_convergence.png` (about 20 seconds), `sweep_grid_alpha_gamma_heatmap.png`
(about 30 seconds), then `controllers_test.png` (about 90 seconds, now with the oracle bar in black).

"I ran seven sweeps, 235 runs, five seeds each, changing one factor at a time. This is the exploration
sweep. A fast epsilon decay converges in about 1,000 episodes, a slow one takes about 2,500, but all end
at about the same quality. So the schedule changes how fast it learns, not where it ends up."

"This heatmap is learning rate against discount. Low learning rates are better in every column. High
learning rates chase the noise in weekly demand. A coarse state grid also beat the fine one, because the
fine grid has only about 90 visits per state."

"Now the final test, on 2010. Doing nothing costs 45 dollars 31 a week. The rule-based controller is
4.9 percent worse, because charging at full power raises the weekly peak and the demand charge punishes
that. The simple greedy rule saves 7.8 percent. Q-learning with the plan's settings is 4.9 percent worse than
doing nothing. My best version, with a coarse grid, saves 1.5 percent plus or minus 1.4, about 70 cents a
week. So in this simulation, Q-learning does not beat the simple greedy rule."

"Now the oracle. With perfect foresight it saves 35.6 percent, about 16 dollars a week. So there is a lot of
room. The greedy rule captures about a fifth of that, my best learner about 4 percent. About 59 percent of the
oracle's saving is flattening the weekly peak: it cuts the peak from 3.45 to 1.55 kilowatts. The other 41
percent is moving energy to cheap hours. That matches what I found in the sweeps: the learners miss the peak. And limiting the oracle to my own five
power levels still saves 27.5 percent. So most of the gap is learning, not the action set."

"Generalization: every controller lost ground from 2009 to 2010, including the ones that never learn, partly
because peak demand fell from 4.3 to 3.5 kilowatts. Summer was hardest: the summer week's whole peak is
under my 3 kilowatt charge power, so full-power charging always creates a new peak."

## 5:30 to 7:00 | Demo and what the agent learned

**Show:** `representative_days.png`, then `policy_heatmap_selected.png`. If you have time, run
`uv run python scripts/run_baselines.py` live (about 10 seconds) to show the numbers appear.

"Here is a winter day and a summer day from the held-out weeks. Black is the real demand. In winter the
full-power rule-based controller pushes the grid import above 5 kilowatts late in the evening, while the
Q-learning agent stays at or below the home's own demand there. In summer the rule-based controller creates a 4 kilowatt spike
that the home never had. The dashed black line is the oracle: it keeps the grid import almost flat, about
0.9 kilowatts all day in summer. That is how you beat a demand charge."

"This heatmap is what the selected agent learned: for each battery level and time of day, which action it
prefers. It is mostly idle, with gentle charging and some evening discharge, and it never charges at full
power. That is why it generalizes best: it does little, so it rarely makes a costly mistake. It also leaves
most of the possible saving unused."

## 7:00 to 7:40 | Limitations and contributions

**Show:** the Limitations list in `docs/GENERALIZATION.md` or the report.

"Limits: one household, one made-up tariff, one test year, and a hand-chosen discretization. I chose the
settings on validation data, so validation numbers are optimistic. The test numbers are the unbiased ones. I
did not tune the simple rules, so the comparison may understate them."

"This was an individual project. [YOU: say in your own words what you did and what you checked.]"

## 7:40 to 8:00 | Wrap-up

"To sum up: on real data with a simulated battery, Q-learning learned to cut energy cost but only slightly
beat doing nothing once the demand charge was included, and a simple greedy rule did better. An oracle with perfect
foresight shows about 36 percent is possible, so the next step is a learner that can see the peak. The code, results and tables are in
the repository. Thanks for watching."

---

## Timing check

| Section | Time | Words (about) |
|---|---|---|
| Problem | 0:00 to 0:45 | 100 |
| Data and environment | 0:45 to 2:00 | 160 |
| Methods | 2:00 to 3:30 | 190 |
| Results | 3:30 to 5:30 | 320 |
| Demo | 5:30 to 7:00 | 180 |
| Limitations and contributions | 7:00 to 7:40 | 80 |
| Wrap-up | 7:40 to 8:00 | 55 |

## Questions you should be ready to answer

- Why did you change the reward? (Noise and exploitation, shown by the `plan_reward` ablation: -9.8 ± 0.8%.)
- Why does the greedy rule beat Q-learning? (It charges gently, so it avoids new peaks. The Q-learner cannot
  see its own peak.)
- Is +1.5% a real win? (It is about one standard deviation from zero. Say so.)
- Why July for summer and not August? (The household was away for much of August.)
- Why is the coarse grid best? (Sparse visits for the fine grid, and the coarse policy is conservative.)
- How good is the oracle as a benchmark? (Both oracles know the whole week. I built a second one limited to my five
  power levels: it saves 27.5% against 35.6% continuous, so the action set explains about a quarter of the gap
  to my best learner. The solver hit its 30-second limit on half the test weeks, so I report the best plan found
  and a proven ceiling 1.3 points higher.)
- Does leftover battery energy change anything? (I checked: valuing it at the off-peak price shifts savings by
  under one point and changes no ranking. Greedy 7.8 to 8.5 percent, my best learner 1.5 to 1.3.)
- What is still missing? (Deep RL and the language-model layer.)
