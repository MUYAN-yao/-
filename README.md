# CIL 科研合作自学包：从看懂问题到负责一个实验

版本：2026-09-09。对象：首次加入本课题的合作者。你不需要先读完整个项目，也不需要训练大模型。

**双击 index.html 开始阅读。** 首轮目标是理解研究问题，并使用既有分数和标签重算论文主表的 IITB 三行。做完后交付解释与问题清单，而不是只提交“运行成功”截图。

## 阅读与执行顺序

1. 阅读 [中文讲解](guide.md)，先看示意图和真实对照结果。不要把示意图当真实视频。
2. 阅读 [首轮任务书](templates/01_首轮任务书.md)，明确你负责什么、不负责什么。
3. 运行玩具示例，再校验输入，再重放原实验。运行方法见下方。
4. 用 [实验记录模板](templates/02_实验记录模板.md) 记录结果；按 [提问与周报模板](templates/03_提问与周报模板.md) 整理疑点。
5. 先独立回答 [自检题](templates/04_自检题与参考答案.md)，再看答案。第二阶段只设计网格敏感性分析，得到导师确认后才执行。

## 环境准备（只做一次）

完整解压 ZIP 到普通文件夹，不能在压缩包预览内运行。推荐 Python 3.12；固定依赖为 NumPy 2.3.5，适合 Python 3.11–3.13。无需 GPU、PyTorch、原始视频、模型权重或数据下载。

在材料包根目录打开终端。建议使用隔离环境：

```text
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\toy_demo.py
.venv\Scripts\python.exe scripts\reproduce.py --check-only
.venv\Scripts\python.exe scripts\reproduce.py
```

如果 Windows 只有 `py` 命令，把第一行改成 `py -3 -m venv .venv`。使用隔离环境时按以上命令运行，不必激活环境。Linux/macOS 将 `.venv\Scripts\python.exe` 换成 `.venv/bin/python`，路径分隔符用 `/`。

如果当前 Python 已有 NumPy，可直接执行：

```text
python scripts/toy_demo.py
python scripts/reproduce.py --check-only
python scripts/reproduce.py
```

Windows 的 `RUN_REPRODUCE.cmd` 只是方便入口，优先使用包内 `.venv` 的 Python，否则使用系统 `py -3` 或 `python`；不自动安装依赖。脚本不会联网；首次安装 NumPy 需要网络或你已有的离线安装包。

## 你会看到什么

- 玩具示例：第一组 AP 都为 1、单对 TIA 为 0；第二组 donor AP 为 5/12。它们是等长网格上的数学例子，不是正式样本。
- 输入检查：核对封存预测、协议、原评估脚本和 150 个标签的哈希；不计算指标。
- 原实现重放：完整三候选、8 个 donor、134 个归属目标、10,000 次按视频抽样。CPU 运行，具体时间依机器而异，不需要反复启动。
- 每次新建 `outputs/replay_时间戳/`：`REPORT.md` 是易读结果，`table_ii_iitb.csv` 是三行表与区间，`comparison.json` 是与封存结果的逐项对比，`environment.json` 记录环境。
- `REPLAY_RESULT.json` 明确标为教学重放。内部历史字段描述原研究，不代表你进行了新的一次独立确认。

本入口调用原评估函数，因此首先是**原实现重放**，不是独立代码复现。要形成科研贡献，还要解释分母、对照、统计单位、结论边界，并经导师审查提出下一步实验。

## 目录说明

|目录/文件|用途|
|---|---|
|index.html / guide.md|自学入口与中文讲解|
|templates/|任务书、记录模板、提问模板、自检题|
|assets/|论文原理图、真实汇总结果图及对应汇总数值|
|paper/|当前论文正文/补充 PDF、投稿前复核报告，仅为稿件快照|
|inputs/|两份固定预测包、协议、对齐说明、原标签及其记录|
|reference/RESULT.json|封存参考答案，只读，不是学生产出|
|scripts/evaluator_original.py|原评估代码的逐字副本，不改原文件|
|scripts/reproduce.py|安全校验、重放、比较、生成报告的教学入口|
|validation/|材料制作者的入口测试记录，不代替你自己的运行和分析|
|outputs/|你运行后新生成的结果；初始包不含作者运行输出|

## 出错怎么办

1. `No module named numpy`：确认安装和运行用的是同一个 Python。优先使用上面的 `.venv\Scripts\python.exe`。
2. 哈希不匹配或标签缺失：不要改代码绕过；重新解压一份完整包，核对是否误编辑了 inputs/reference。
3. `DIFFERENCE DETECTED`：保留全部输出和环境信息，按模板提问。不要换种子或阈值把数值调到一致。
4. 卡住：记录启动时间与终端输出，先确认仍在计算。窗口关闭前保存报错，不要并行启动多份相同实验。

## 数据与合作边界

先阅读 [数据和使用说明](DATA_NOTICE.md)。本包不含原始视频、权重、密码或服务器凭据。只用于导师确认有权共享的课题组内部研究；不得默认上传公开仓库或云端 AI。

问题按“我想检验什么—做了什么—观察到什么—我的解释—需要确认什么”提问。不要只发截图说“跑不通”。署名根据实际贡献讨论，本包不承诺论文录用或作者排序。
