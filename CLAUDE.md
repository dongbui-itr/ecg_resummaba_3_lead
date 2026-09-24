# ecgr - working notes for Claude

Start with the skill `ecgr-overview` (`.claude/skills/ecgr-overview/SKILL.md`); it routes to
the specialised skills (`ecgr-data-build`, `ecgr-train-sweep`, `ecgr-evaluate`,
`ecgr-model-family`, `ecgr-label-free`, `ecgr-tests-guards`).

Contract: input `(15000, 3)` = 60 s, 3 leads, 250 Hz, annotated lead on channel 0. Outputs
`beat_cls (3000, 4)` softmax over None/N/V/S per 20 ms step and `lead_quality (3000, 3)`
sigmoid per lead (argmax of its time-average = the most reliable channel). Labels count only
inside the reviewed span (`IGNORE_LABEL = 255` elsewhere). Four sizes: `resumamba_5m/3m/1m/100k`.

Python: `/home/ai-server/miniconda3/envs/beat/bin/python` (system python has no TensorFlow).
Tests: `./run_pipeline.sh test`. Never train on EC57 databases; never tune on mitdb or the
beat-eval set - calibrate on portal-eval; diff every new checkpoint against
`assets/baselines/10s_3lead/` with `python -m ecgr regress`.
