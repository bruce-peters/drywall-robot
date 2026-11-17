# CLAUDE.md — drywall-robot

MuJoCo simulation environment for robotic drywall panel placement (take-home, Part A).

## Working rules for this project

- **Lab notebook:** log everything you do in [LABNOTEBOOK.md](LABNOTEBOOK.md) — commands run,
  findings, design decisions, failures and their fixes. Append, newest at the bottom.
- **Design decisions go to chat first.** Propose any design decision to the user in chat and
  get an answer before implementing it; then record it in LABNOTEBOOK.md tagged **[DECISION]**.
- Keep it small, correct and reproducible. Don't add features that weren't asked for. If
  something fails after a couple of tries, simplify and document the simplification in the
  README instead of going in circles.
- Build order: (1) arm loads + IK tracks a moving target; (2) panel + attach/release;
  (3) scripted placement on one seed + video; (4) randomization, reward terms, tests, README.
