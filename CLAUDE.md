# Home Energy RL Agent: rules for Claude Code

Project: RL smart energy management. Spec is in PROJECT_PLAN.md. Follow it.

## Hard rules
- Python 3.11, uv for deps, ruff for lint, pytest for tests. Seed everything (default seed 42).
- All experiments run from YAML configs in configs/. No magic numbers in code.
- Every result number in README/report must come from a real run saved in results/. Never invent or estimate numbers. If a result is missing, say so.
- Label simulated vs real clearly everywhere: demand is REAL (UCI dataset); battery, tariff, and any PV are SIMULATED.
- Comment each major section of every script (what it does and why). Keep functions small.
- For every module, also write a 10-15 line plain-English entry in docs/EXPLAINER.md so the student can explain it on video.
- Do NOT add AI-usage markers: no AI-ASSISTED header comments, no docs/AI_USAGE.md.
- Never commit raw data, API keys, or .env. Data is downloaded by scripts/download_data.py.
- The app and agent must work without an API key via a clearly marked offline fallback.
- Commit after each phase with a clear message. Run `pytest` and `ruff check` before committing.
- Commits have no Claude co-author line. Never push; give the user the push command.
