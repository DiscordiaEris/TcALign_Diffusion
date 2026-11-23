# Face editing training repair implementation plan

**Goal:** Correct demonstrated causes of conditioning collapse, make evaluation reflect editing quality, and launch a verified fresh training pipeline in a detached server session.

**Scope and authorization:** User explicitly requested diagnosis, fixes, and background server training. Work in the existing checkout: all prior implementation is uncommitted and model/data assets live here. Preserve unrelated edits and original checkpoints/data. No architectural replacement or new pretrained dependency.

## Evidence and decisions

- Adam's coupled L2 decay destroys small pretrained attention weights: 400 zero-task-gradient updates at the existing settings erase 97.3% of a 0.02 weight; actual checkpoint attention norms retain only a few percent of pretrained norms. Use AdamW, exclude bias/norm vectors from decay, use a smaller LR for pretrained UNet parameters; reset Stage2 from local SD weights rather than resume damaged attention.
- Same-identity FEI pairs are valid. Pose is 73% of usable training pairs; use opt-in instruction-balanced sampling. Occasional expression/lighting co-variation is dataset noise, not a globally wrong lighting reference. Do not rewrite capture labels.
- Stage1 pose predictions distinguish directions. Audit all holdout identities before deciding on retraining; fresh Stage1 training with corrected optimizer/sampling avoids carrying damaged parameters forward.
- Fixed four-example sampling ablation: full-noise identity cosine can be near zero; strength .55 recovers identity but mostly copies the source. Preserve explicit strength controls and checkpoint their defaults. This mitigation is not evidence that edit control is solved.
- Validation currently resamples noise/timesteps and auxiliary availability every epoch. Use a fixed validation RNG stream while restoring training RNG, log generated identity/control as well as SSIM/PSNR and source-copy baseline.
- Zero-delta mapper is exactly identity, so CFG unconditional bypass is correct and needs no patch.

## Work

1. Optimizer: failing regression for small pretrained weights, parameter coverage, separate pretrained LR and decay exclusions; implement in BaseTrainer; save the quantitative audit report.
2. Sampling: failing sampler tests for equal instruction probability and repeatability; add opt-in sampler in dataset and both entrypoints.
3. Inference configuration: failing tests for saved/restored strength and BF16 metadata and omitted strength defaults; wire config, checkpoint loader, generate/CLI/evaluate/validation consistently. Explicit CLI wins.
4. Validation: failing tests for reproducible validation and unchanged outer RNG; fixed seeded context, component means/counts, generated FaceNet identity, Deep3D requested attribute errors, and source-copy baseline.
5. Regression suite; fresh pretrained GPU smoke: forward/backward, finite gradients, checkpoint save/reload, sampling. Review scoped diff and address substantive findings.
6. New versioned configs/logs/checkpoints and launch script; snapshot exact source/configs; archive old latest checkpoint and terminate only verified old trainer process after repair validation; start detached tmux pipeline Stage1 → Stage2. Verify persistent process, actual optimizer updates, and checkpoint creation.

## Completion limits

Code-level regression/smoke checks must pass before launch. Full visual quality/control improvements require new learning and holdout evaluation; do not promise they are solved by a few steps. Keep old reports and runs for comparison.
