# Identity and exposure repair

Pause the old trainer, retain checkpoint190750 and the completed Stage1 editor, fix identity drift and dimming, verify final generated images, then restart background Stage2 fine-tuning.

Evidence: recursive sampling undoes dimming, raw FaceNet penalizes very dark same-identity targets, and FEI file numbers13/14 do not reliably encode two darkness levels. Source image channels alone give weak identity retention through the denoising trajectory.

Implementation: frozen pretrained IP-Adapter-FaceID with matching buffalo_l embeddings and companion LoRA; preserve LoRA separately from trained base weights for safe loading/resuming. Normalize exposure in the identity loss. Replace contradictory lighting pairs in a new dataset with controlled same-image exposure pairs, preserving identity splits and all nonlighting examples. Fit a small text/source-exposure head on training identities with an internal holdout; calibrate final RGB exposure only for explicit lighting requests. Train and inference share this path; pixel exposure supervision replaces the ambiguous Deep3D lighting proxy when this head is enabled.

Validation gates: unit tests, frozen-LoRA/checkpoint equivalence, finite full-model training updates and saved checkpoint reload, source detection coverage, six matched held-out identities × five edits with fixed noise, novel lighting wording, before/after independent identity/exposure/pose metrics, visual inspection, then isolated tmux training with source snapshot and dependency hashes.

Completed:67 tests;1,578 source faces detected;64-step GPU shortfit/reload;30 matched edits passed. Independent pose identity cosine .494→.671, expression .681→.780, exposure error .650→.00195, yaw error5.198°→5.151°,12/12 directions correct. Identity drift remains measurable. Global exposure calibration does not claim physical directional relighting.

The final training path uses ordinary diffusion updates at lower learning rates. Full-sampler backpropagation was rejected because its shortfit softened facial/hair detail without improving independent identity metrics.
