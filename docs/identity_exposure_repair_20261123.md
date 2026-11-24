# 身份与曝光修复（2026-11-23）

旧训练在190750步停止，保留原权重与原始数据。新配置为 `configs/runs/fei_stage2_identity_exposure.yaml`，数据目录 `datasets/face_editing_exposure_20261123`，来源与逐对曝光系数记录在其中的 `manifest.json`。训练集保留3960个非光照配对，以1383个同图曝光配对替换1080个不可靠的光照配对；验证身份保持隔离。受控配对是全局RGB曝光变化，不是真实方向性重光。

身份条件使用[官方 IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID)及配套 InsightFace buffalo_l 的 `w600k_r50.onnx`；不会用 FaceNet/SFace 向量替代。适配器与伴随LoRA冻结，LoRA以独立残差加载，避免覆盖已训练权重或续训重复融合。冻结权重不打包进训练checkpoint；运行目录的 `input_sha256.txt` 固定外部权重及数据清单。曝光归一化身份损失解除身份和暗光的监督冲突。

曝光头只输入源图曝光和冻结文本特征，目标亮度仅用于训练监督。162个训练身份拟合，18个训练身份内部选权重；20个原验证身份隔离。全局输出gain只用于明确光照指令，不修改未请求的输出。它修正最终亮度，不能替代复杂阴影、方向性照明或3D光照建模。

同样6个持出身份、每人5项编辑、固定噪声的30项对照：

| 指标 | 暂停前 | 修复并短训64步后 |
|---|---:|---:|
| 转头SFace身份相似度 |0.494|0.671|
| 转头FaceNet身份相似度 |0.703|0.777|
| 表情SFace身份相似度 |0.681|0.780|
| 亮度归一化暗光SFace |0.778|0.820|
| 光照曝光误差（越低越好） |0.64969|0.00195|
| 转头平均yaw误差 |5.198°|5.151°|
| 转头方向正确 |12/12|12/12|

指标改善不等于身份完全不变；应持续检查皮肤、头发与面部细节。完整采样反传方案的短训曾使细节变软，已从最终训练路径移除。

检查：67项单元测试通过，全部1578个源人脸检测成功，真实GPU64步训练和checkpoint重载通过，独立审查未发现阻断问题。最终结果与图像在 `outputs/identity_light_fix_20261123/repaired_shortfit_validation`。

后台入口：`scripts/train_fei_identity_exposure_server.sh`。tmux socket `tcadiff-identity`，session `fei-identity-exposure`。日志：`logs/fei_identity_exposure_20261123/console.log`；checkpoint：`checkpoints/fei_stage2_identity_exposure_20261123/latest_checkpoint.pth`。训练从已验证的64步权重初始化，重置优化器和本轮步数，保留完成的Stage1。

复查命令：

```bash
LD_PRELOAD=/usr/local/lib64/libstdc++.so.6 OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -q
LD_PRELOAD=/usr/local/lib64/libstdc++.so.6 HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.evaluate_identity_exposure --config configs/runs/fei_stage2_identity_exposure.yaml --checkpoint checkpoints/fei_stage2_identity_exposure_20261123/latest_checkpoint.pth --output outputs/identity_light_fix_20261123/recheck
```
