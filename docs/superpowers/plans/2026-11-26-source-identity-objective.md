# Source identity objective repair plan

Goal: resolve the observed identity-loss/evaluation disagreement, validate actual generated edits, and restart background training.

Evidence: best vs late full-sampling outputs on 12 fixed edits show SFace and buffalo_l ArcFace both decrease, while FaceNet source identity increases. Holding detections fixed does not remove the SFace decline; using the same crop in both recognizers does not remove the disagreement. FaceNet gradients are nonzero, RGB is correct, target/source gradients are mostly aligned, and exposure normalization has its intended null direction. These are not proven implementation bugs; the primary failure is a proxy objective that does not transfer across recognition models.

Choice: use frozen local buffalo_l ArcFace with source-only identity and its canonical five-point RGB crop; preserve SFace as the independent evaluator. Merely changing target weights or freezing landmarks is unsupported by the causal tests. Using SFace itself as the training loss would remove the independent check and is excluded.

- [x] Stop-state/process and best/latest checkpoint audit; preserve previous work and checkpoints.
- [x] Isolate crop, reference and exposure effects, save raw generated tensors and scores.
- [x] TDD: frozen differentiable ArcFace graph, canonical crop, finite image gradients, target independence, detection fallback reporting.
- [x] Add explicit backend config preserving legacy defaults; wire model identity loss dispatch.
- [x] Report the correct recognition backend in validation and keep independent SFace selection/control gates unchanged.
- [x] Verify ONNX/PyTorch forward parity, real GPU full-trajectory gradients, frozen-backbone equality and checkpoint reload.
- [x] Pilot on training identities only; compare full held-out panel and a second cohort/seed with the existing best. Require independent identity improvement and no control regression before production launch.
- [x] Run complete CPU suite and independent code review, then launch the new server background job and verify actual updates/checkpoint persistence.

Only the existing 16 bounded identity gains may train. Generation backbone, source conditioning, geometry and exposure weights remain fixed. Use the existing 100-epoch ceiling, 32 full-trajectory updates per epoch, five unsuccessful validation epochs trigger early stopping. Do not claim complete identity invariance from a finite panel or a small gain.

Completed with a rank2 private identity output refinement after 16-gain controls failed. Production uses the validated generated-alignment profile; paired_target remains an explicitly tested alternative which did not beat the independent primary score. See docs/identity_metric_repair_20261126.md for measured limits and startup/checkpoint evidence.
