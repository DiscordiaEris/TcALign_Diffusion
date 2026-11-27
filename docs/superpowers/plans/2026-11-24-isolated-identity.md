# Isolate pretrained identity projections from text fine-tuning

User requested stop, diagnose identity drift, repair, then resume background training. The previous 20-epoch setting was an explicit repair fine-tuning configuration; the new run restores the original 100-epoch ceiling.

Training PID312718 stopped after archiving step92650. Original weights, data, and completed Stage1 remain available. Do not publish changes without a separate request.

Evidence: no cross-identity pairs or train/val identity overlap; identity gradients and alignment/transport math are connected/correct. FaceID projection and LoRA match the official numerical formula. However IP identity attention shares trainable native Q and output projections. Identity-only backward creates Q/out gradients despite frozen IP K/V and LoRA. Relative warmstart-to-late Q/out changes are1.48%/2.64%, with some high-resolution output layers changing13.6%. Single-step identity loss also does not supervise the complete generation trajectory.

Controlled GPU ablation on the same four held-out head turns: original mean SFace .632, disabling identity attention .373, doubling shared identity scale .713, private pretrained Q/out at normal scale .680, private Q/out with scale2 .756. Restoring shared native Q/out to the SD base damaged pose control, so retain trained text attention and isolate only identity projections. Geometry/mapper removal did not repair identity and impaired edits.

Implementation: FrozenIdentityAttnProcessor computes text with current attention, identity with private frozen Q/out including pretrained companion LoRA. Construct private projections from the unchanged SD+FaceID base before loading trainable checkpoint weights. This makes rebuild/resume deterministic without depending on unsaved fine-tuned frozen tensors. Bias/residual/rescale are applied once; hidden-state gradients remain connected.

Monitor actual final outputs: six fixed held-out identities times left/right/smile/dim/very-dim; independent SFace identity only on pose/expression; complete detection coverage; pose direction and mean error plus exposure gates before replacing best_identity_checkpoint. Store score state and reconcile with the on-disk best after both supported resume paths. Legacy initial weights receive a baseline evaluation without perturbing training RNG.

Validation gates: CPU original-processor output and gradient equivalence; optimizer isolation; metadata roundtrip; detector/pose/exposure coverage and rejection gates; initial/best score restoration and resume ordering; full suite; real64-step training and reload; matched six-identity30-edit before/after generation; inspect images; only then launch isolated tmux training with100-epoch ceiling, snapshots and dependency hashes. Keep the validated best initial checkpoint even when a later latest checkpoint scores lower.
