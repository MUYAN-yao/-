# 数据、来源与使用边界

本包是现有 CIL 研究的内部教学与合作材料，不是公开数据发布包，也不是已经投稿或录用的论文附件。

- 标签来源于已使用的 IITB Corridor 测试端点，分数来自本项目封存的三个候选。原论文作者与数据集作者不是同一概念。本包不授予新的数据许可证。
- 仅在导师确认接收方具有相应研究访问/共享权限后，向该合作者提供本包。未获许可者应先解决访问权限，不能把“教学”当作任意再分发授权。
- 不上传 labels、预测 ZIP、未发表稿件、实验明细到公共仓库、网盘公开链接或云端 AI。AI 可协助解释公开概念、分析脱敏报错、实现玩具代码；涉及研究资料时先确认允许的工具与范围。
- 不含原始视频、加密档案密码、服务器密钥、模型权重或无关项目材料；首轮重算不需要这些内容。
- inputs、reference 和 evaluator_original.py 是只读参考；新的分析写入 outputs 或单独的学生工作目录，不覆盖历史记录，不重新打开任何所谓“一次性标签库”。
- 原始研究的“一次打开标签”和“独立确认”字段仅描述历史来源。学生已知结果后的运行不能获得同等证据身份。
- 图 cil_protocol.png 中的曲线是数学示意；cil_ownership_control.png 是合成视频数据集 Pistachio 上的真实汇总实验结果，不是 IITB 的单个视频。
- PDF 和复核报告为随包附带的稿件快照；其中作者待办、历史声明及投稿状态不代表本包已补全或已提交论文。追加任务是研究提案，不是预先保证存在正结果。

## 来源定位

原评估实现：项目 rLEM-VAD/scripts/evaluate_iitb_corridor_prospective_cil_v1.py。

IITB 封存端点：logs/cil_acceptance_lift_v2_20260824/cycles/CYCLE-B01_iitb-corridor-prospective-cil/。

论文快照：paper/cil_tmm_v1/。

官方数据来源链接保留在 inputs/SCIENTIFIC_CONTRACT_V1.json 的 official_page/paper 字段。原协议是来源记录，不是新的共享许可证。package_manifest.json 为随包文件身份与来源清单，哈希只能核对完整性，不能证明公开时间戳或数据权利。
