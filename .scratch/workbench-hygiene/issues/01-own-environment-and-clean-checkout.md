# 01: Its own Python environment; the 9 ad-hoc scripts committed or archived

**What to build:** The pipeline's interpreter is a venv under `$REMOTE_ROOT/envs/` built from `pipeline/requirements.txt`, not MILo's conda environment; and the 9 untracked November 2025 scripts at the checkout are either committed (useful) or archived (not).

**Answers:** none (workbench routine).

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

Amendments 3.4 and 2.7. `AGENTS.md` itself calls the borrowed interpreter fragile: a rebuild of MILo's environment silently changes this pipeline's runs. The conservator agreed the scripts' fate is decided here (2026-10-24).

- [ ] venv built in a short login-node job; `environment.python_interpreter` points at it
- [ ] One stage re-run with it gives the same output as before
- [ ] Each of the 9 scripts committed or archived, listed here
