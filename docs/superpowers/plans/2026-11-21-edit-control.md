# Source-conditioned editing repair

User explicitly requests stop, repair text control, verify, and restart server background training. Old trainer stopped, checkpoint archived. Keep architecture, existing pretrained assets, and prior runs.

Evidence: .65 source-start sampling ignores pose even with oracle geometry; pure-noise sampling correctly turns left/right but loses identity. Teacher-forced target noising trains a different input than source-start sampling. Geometry projection/sign correct; target-pair landmark frames shift systematically ~40–60 px for pose photos. Lighting 3D supervision assigns >80% of its copy penalty to unrequested attributes.

1. Regression then add mixed source-start transport: half conditional examples start with noised source at t250..700 and regress effective epsilon relative to clean target. Retain standard target diffusion otherwise. Exclude CFG-dropped source transport. Test epsilon/v x0 inversion and identical source inputs/different target labels.
2. Regression then register target photos in the source landmark frame (opt-in). Preserve cached 3D coefficients, return source alignment for registered targets. Validate transform composition/sign.
3. Regression then add requested-attribute displacement-normalized 3D control; ignore target nuisance attributes; extend image auxiliaries through t700. Identity stays active. Explicitly test copied-source penalty and corrective gradient directions.
4. Use healthy repaired20epoch weights, reuse completed Stage1, reset optimizer/epoch in a separate run. Unit/full suite; GPU controlled small-training experiment and heldout same-source/different-prompt ablations before long training.
5. Review; source snapshot, locked detached launcher, actual-update/checkpoint checks. Do not claim final holdout quality from unit tests or cherry-picked overfit samples. Preserve baseline and all experiment metrics.

Follow-up evidence:400-step32-pair/eight-training-identity experiment produced correct left/right on two unseen identities with shared noise (4/4 direction). Lighting coefficients improved but visible exposure barely changed; added and regression-tested aligned-face luminance supervision (six tests), then continued200steps as a focused test. Full suite40passes; code review reports no P1/P2. Strength/identity and smile quality remain separate empirical limitations, not hidden by pose success. Long-run launcher initializes healthy pre-experiment weights, not the small-subset checkpoint.
