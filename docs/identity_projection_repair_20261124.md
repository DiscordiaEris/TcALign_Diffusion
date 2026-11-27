# 身份投影隔离修复（2026-11-24）

旧PID312718按用户要求停止，92650步权重归档于 `checkpoints/fei_stage2_identity_exposure_20261123/stopped_identity_drift_20261124.pth`。此前20轮来自修复微调YAML，不是原项目默认100轮。本次新配置 `configs/runs/fei_stage2_identity_isolated.yaml` 恢复100轮上限。

根因：IPAdapter的K/V、projection和伴随LoRA冻结，但身份注意力仍共用可训练的native Q和to_out，因此文本训练直接改变了身份分支。CPU反向和权重漂移测量证实此路径存在。真实配对没有跨身份/训练验证泄漏；FaceID接线、LoRA公式、CFG零条件、FaceNet梯度与对齐没有发现确定错误。单步teacher损失无法直接代表最终生成身份，这是监控上的另一个缺口。

新处理器把身份Q/out独立冻结，并在载入可训练checkpoint之前从SD+FaceID基础权重构造。保留已训练文本Q/out，避免直接恢复原始SD投影导致转头控制丢失。私有分支保留对输入hidden states的梯度，冻结分支本身不能被优化器修改。冻结依赖仍由本地基础权重重建，架构标记随checkpoint保存。强度由1校准到2；单纯加大旧共享分支强度的消融也有收益，但不及隔离后的结果。

64步真实短训、重载以及相同6人30项编辑、相同噪声对照结果：

| 指标 | 停止时92650步 | 隔离投影并短训后 |
|---|---:|---:|
| 转头独立SFace相似度 |0.6803|0.7500|
| 表情独立SFace相似度 |0.7560|0.8034|
| 转头FaceNet相似度 |0.8090|0.8047|
| 转头yaw平均误差 |4.84°|5.11°|
| 转头方向正确 |12/12|12/12|
| 暗光曝光误差 |0.00383|0.00455|

独立身份指标改善，FaceNet总体基本持平；身份尚未完全保持，头发/衣服细节仍有生成误差，不能宣称完全解决。对照图 `outputs/identity_drift_20261124/comparison_003.png` 为原图、停止权重、新权重三排；逐项原始数据见同目录 before/after/rows.json。

新增每轮30项固定生成验证，身份选择只统计pose/expression18项；独立识别必须完整覆盖、12次pose方向均正确、平均yaw误差≤8°、曝光误差≤0.1且身份均值更高，才更新 `best_identity_checkpoint.pth`。默认保留已经验证的初始best。CLI/YAML两种恢复路径均在加载后与磁盘best阈值取最大，避免旧latest覆盖更好best。legacy初始化先测真实生成基线，不推进训练RNG。

监控固定集采用排序后的前6个完整身份（含046）；最终历史匹配集采用原先已建立baseline的6个身份（含067），两者分开报告。新监控集初始独立SFace18项均值0.7737、yaw误差5.49°、曝光误差0.01217，覆盖及方向均完整。

测试87项通过，包含原IP输出/梯度等价、优化器修改文本投影后的身份残差不变、checkpoint标记、质量门槛/覆盖、两条恢复路径、RNG保护。真实64步损失和梯度有限，GPU重载采样验证完成。

后台入口 `scripts/train_fei_identity_isolated_server.sh`；tmux socket `tcadiff-identity-isolated`，session `fei-identity-isolated`。日志 `logs/fei_identity_isolated_20261124/console.log`。最新进度权重用于恢复：`checkpoints/fei_stage2_identity_isolated_20261124/latest_checkpoint.pth`；推荐生成使用保留的 `best_identity_checkpoint.pth`。新微调重置优化器及轮数，继承完成Stage1和已验证64步权重。运行目录保存源码快照、base commit、diff、依赖SHA及初始检查点路径。
