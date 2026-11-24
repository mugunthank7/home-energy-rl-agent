# Video script (up to 8 minutes)

Spoken at a calm pace of about 130 words a minute (about 1,000 spoken words in total). Everything in
quotation marks is meant to be said out loud; edit it so it sounds like you. Numbers come from
`docs/REPORT_DRAFT.md`. If you re-run any experiment, check the numbers again.

**Setup before recording:** open the repo, the README, and these files in tabs so you can switch quickly:
`reports/figures/eda_daily_profile.png`, `env_check_episode.png`, `controllers_test.png`,
`deep_rl_curves.png`, `representative_days.png`.

**The story in one line:** tabular Q-learning fails, deep RL succeeds, an ablation shows part of why, and an
oracle shows how much is still possible.

---

## 0:00 to 0:45 | Problem and why it matters

**Show:** `eda_daily_profile.png`.

"Homes that pay time-of-use prices and a demand charge can save money with a battery. Charge when power is
cheap, discharge when it is expensive, and avoid setting a high weekly peak. This is the daily demand of one
real French household. It peaks at about 7 in the morning and again from 7 to 9 in the evening, when my
simulated price is highest. The question: can reinforcement learning learn a battery controller that beats
simple rules, and does it hold up on a later year and other seasons?"

"One thing to be clear about from the start: the demand is real data. The battery, the prices and the demand
charge are simulated, so every saving I show is a simulation result."

## 0:45 to 2:00 | Data and environment

**Show:** the README real-vs-simulated box, then `env_check_episode.png`.

"The data is the UCI household power dataset, about two million minute readings. I turned missing values into
gaps, filled gaps up to 3 hours, averaged to hourly, and dropped 24 days that still had holes. I split by time:
2007 and 2008 to train, 2009 to validate, 2010 to test, never shuffled."

"The environment is a Gymnasium environment. One step is an hour and one episode is a week. The battery is 10
kilowatt-hours with a 3 kilowatt limit and 90 percent round-trip efficiency. There are five actions: discharge
high or low, idle, charge low or high. This plot shows one week: demand, grid import, battery level and price.
The tests check that energy is conserved, the charge level stays in bounds, and results repeat under the same
seed."

## 2:00 to 3:30 | Methods

**Show:** the README controller list, or `docs/EXPLAINER.md`.

"I compare these controllers. No battery is the reference. The rule-based one charges at full power overnight
and discharges in the evening. The greedy one uses price percentiles, charges gently, and scales discharge to
demand."

"Then three learners. Tabular Q-learning keeps a table over four summary features: battery level, hour, demand
level and tariff period. PPO and DQN are deep RL: small neural networks that see the full observation, including
the highest grid import so far this week. All learners use the same environment, actions, reward and training
budget, and I did not tune the deep RL settings."

"I changed the reward, and I want to explain why. The plan's peak penalty trained a bad agent: it was noisy and
the agent could not see the peak. I now charge only the peak the battery adds on top of the home's own peak.
That does not change the best policy, but it removes a lot of noise."

"Finally, two oracles for an upper bound. One knows the whole week of demand and prices in advance and uses
continuous power. The second is limited to my five power levels. Replaying its plans through the real
environment reproduces their cost exactly, with zero violations."

## 3:30 to 5:45 | Results

**Show:** `controllers_test.png` (stay on it for most of this section), then `deep_rl_curves.png` for about
20 seconds.

"Here is the final test on 2010, a year none of the agents trained on. Doing nothing costs 45 dollars 31 a week.
The rule-based controller is 4.9 percent worse than that: charging at full power raises the weekly peak, and the
demand charge punishes it. The simple greedy rule saves 7.8 percent."

"Tabular Q-learning struggles. With the plan's settings it is 4.9 percent worse than doing nothing. My best
tabular version, with a coarse state grid, saves 1.5 percent, plus or minus 1.4, about 70 cents a week. I
also swept the learning rate, discount, exploration, training length, battery size and grid resolution, 250
runs in all. Coarser grids and lower learning rates helped, but nothing got it near the greedy rule."

"Deep RL is a different story. DQN saves 10.5 percent, plus or minus 2.2, and PPO 7.1 percent. DQN beats the
greedy rule on average, four of its five seeds, and PPO roughly matches it. Both also cut the weekly peak to
about 2.9 to 3 kilowatts, below the 3.45 with no battery, which no other non-oracle controller does. And they
stay positive in summer, where the rule-based controller loses 20 percent."

"Why? One reason is that they can see the peak. When I hide that input and retrain, DQN falls from 10.5 to 3.4
percent and its weekly peak goes back up. PPO falls less, from 7.1 to 5.8. And giving the same peak
information to the table did not help. I think a network can generalize across similar situations where a table
runs out of data, but I did not isolate that. It also costs about 30 times more training time."

"Now the oracle. With perfect foresight it saves 35.6 percent. Limited to my five power levels it still saves
27.5. So DQN captures about 38 percent of what the five-level oracle can do, PPO 26, the greedy rule 28, and
the best table about 6. About 59 percent of the oracle's saving is flattening the peak. That is what the table
misses."

## 5:45 to 7:00 | Demo and generalization

**Show:** `representative_days.png`. If you have time, run `uv run python scripts/run_baselines.py` live (about
10 seconds) to show the numbers appear.

"Here is a winter day and a summer day from the held-out weeks. Black is the real demand. In winter the
full-power rule-based controller pushes grid import above 5 kilowatts late in the evening. In summer it
creates a 4 kilowatt spike the home never had. The dashed line is the oracle: it keeps the grid import almost
flat, about 0.9 kilowatts all day in summer. That is how you beat a demand charge."

"Every controller lost ground from 2009 to 2010, including the ones that never learn, partly because peak
demand fell from 4.3 to 3.5 kilowatts. Summer is hardest for anything that charges at full power, because the
summer week's whole peak is under my 3 kilowatt charge limit."

## 7:00 to 7:40 | Limitations and contributions

"Limits: one household, one made-up tariff, one test year. The deep agents see more than the table, so this is
not a pure comparison of algorithms. I did not tune any learner, and DQN varies a lot between seeds, from 7 to
13 percent. The oracle needs the future, so no real controller can reach it. Training and test data are
separate: I fixed every setting before looking at 2010."

"This was an individual project. [YOU: say in your own words what you did and what you checked.]"

## 7:40 to 8:00 | Wrap-up

"To sum up: on real data with a simulated battery, a table-based learner barely beats doing nothing, but deep
RL beats or matches a simple price rule and cuts the weekly peak, with the peak input explaining a large part
of DQN's edge. Even so, a perfect-foresight plan saves about three times as much, so there is plenty of room
left. The code, results and tables are in the repository. Thanks for watching."

---

## Timing check

| Section | Time | Words (about) |
|---|---|---|
| Problem | 0:00 to 0:45 | 100 |
| Data and environment | 0:45 to 2:00 | 140 |
| Methods | 2:00 to 3:30 | 200 |
| Results | 3:30 to 5:45 | 330 |
| Demo and generalization | 5:45 to 7:00 | 150 |
| Limitations and contributions | 7:00 to 7:40 | 100 |
| Wrap-up | 7:40 to 8:00 | 75 |

That is about 1,100 words, a little over the 8-minute pace. If you run long, cut the sweep sentence in the
results and the live demo.

## Questions you should be ready to answer

- Why did you change the reward? (Noise and exploitation, shown by the `plan_reward` ablation: -9.8 ± 0.8%.)
- Why does deep RL beat the table? (Two differences: it sees more, including the peak so far, and a network
  generalizes across similar situations. The ablation shows seeing the peak matters a lot for DQN, a little for PPO.
  I did not isolate function approximation from the other inputs.)
- Is the comparison fair? (Same environment, actions, reward, sample budget and evaluation weeks; no tuning on
  validation or test. But the deep agents see the full observation and the table sees 4 features, by the plan's
  design.)
- Is DQN's win over the greedy rule real? (About 2.7 points on average, roughly one seed standard deviation. Four of
  five seeds beat the rule. PPO is level with it. Call it an edge, not a rout.)
- Does leftover battery energy change anything? (I checked: valuing it at the off-peak price shifts savings by
  under one point and changes no ranking. DQN 10.5 to 10.8 percent, greedy 7.8 to 8.5.)
- How good is the oracle as a benchmark? (Both oracles know the whole week. The second is limited to my five power
  levels: 27.5% against 35.6%. The solver hit its 30-second limit on half the test weeks, so I report the best plan
  found and a proven ceiling 1.3 points higher.)
- Why July for summer and not August? (The household was away for much of August.)
- What is still missing? (The language-model agent layer, tuning the learners fairly, and other tariffs or years.)
