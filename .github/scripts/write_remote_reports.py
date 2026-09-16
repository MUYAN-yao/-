from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports"

DISPLAY_NAMES = {
    "resnet18_normal_diagonal": "正常特征距离",
    "resnet18_temporal_novelty": "时序新颖度",
    "pixel_frame_difference": "相邻帧差分",
}

SENTINELS = (
    "sentinel_center_prior",
    "sentinel_duration_center",
    "sentinel_linear_progress",
    "sentinel_sampling_grid",
)


def fmt(value: object) -> str:
    return f"{float(value):.6f}"


def latest_replay_dir() -> Path:
    folders = sorted((ROOT / "outputs").glob("replay_*"))
    if not folders:
        raise SystemExit("No replay output directory was created.")
    return folders[-1]


def main() -> None:
    folder = latest_replay_dir()
    envelope = json.loads((folder / "REPLAY_RESULT.json").read_text(encoding="utf-8"))
    result = envelope["recomputed_historical_result"]
    comparison = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
    environment = json.loads((folder / "environment.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((folder / "table_ii_iitb.csv").open(encoding="utf-8-sig")))

    audit = result["dataset_audit"]
    candidates = result["candidate_results"]
    ordinary = result["ordinary_detection_metrics"]
    sentinels = result["sentinel_results"]
    date = dt.date.today().isoformat()
    REPORT_DIR.mkdir(exist_ok=True)

    candidate_lines = []
    for row in rows:
        key = row["candidate"]
        candidate_lines.append(
            f"| {DISPLAY_NAMES.get(key, key)} | `{key}` | {fmt(row['tia'])} | "
            f"[{fmt(row['tia_lower'])}, {fmt(row['tia_upper'])}] | "
            f"{fmt(row['tia_simultaneous_lower'])} | {fmt(row['cnm'])} | "
            f"[{fmt(row['cnm_lower'])}, {fmt(row['cnm_upper'])}] | "
            f"{fmt(row['cnm_simultaneous_lower'])} | {fmt(row['auc_all_150'])} | "
            f"{fmt(row['ap_all_150'])} | {row['qualified']} |"
        )

    sentinel_lines = []
    for key in SENTINELS:
        item = sentinels[key]
        sentinel_lines.append(
            f"| `{key}` | {fmt(item['point'])} | "
            f"[{fmt(item['ci95_lower'])}, {fmt(item['ci95_upper'])}] |"
        )

    ordinary_lines = []
    for key in result["candidate_family"]:
        metric = ordinary[key]
        checks = candidates[key]["checks"]
        ordinary_lines.append(
            f"| {DISPLAY_NAMES[key]} | {fmt(metric['frame_auc'])} | "
            f"{fmt(metric['frame_ap'])} | {fmt(metric['ap_above_prevalence'])} | "
            f"{'通过' if all(checks.values()) else '未全部通过'} | "
            f"{'通过' if candidates[key]['qualified'] else '未通过'} |"
        )

    mismatch_note = (
        "在相对容差 1e-10、绝对容差 1e-12 下，嵌套结果字段与参考结果一致。"
        if comparison["matches_reference"]
        else f"发现 {len(comparison.get('mismatches', []))} 项差异；应保留并逐项解释。"
    )

    record = f"""# CIL 首轮实验记录

## A. 基本信息

- 记录 ID / 日期：`remote-replay-{date}`
- 执行者：Codex 协助执行；学生署名待补
- 本次状态：原实现重放
- 本次问题：保存预测、150 个标签、冻结协议和原评估实现能否重放 IITB 三候选主表及四个哨兵？
- 导师确认记录：原实现重放不需要追加实验确认；第二阶段网格敏感性分析未确认，不执行。
- 输入包版本 / 校验清单：GitHub 仓库 `main`；`scripts/reproduce.py --check-only` 通过。
- Python / NumPy / 操作系统版本：Python {environment['python'].split()[0]}；NumPy {environment['numpy']}；{environment['platform']}。
- 代码副本路径与哈希：`scripts/reproduce.py`；`scripts/evaluator_original.py`；哈希见 runner 生成的 `environment.json`。
- 本次输出目录：GitHub Actions runner 临时目录 `{folder.as_posix()}`；未提交原始 `outputs/` 明细。
- 命令原文与工作目录：仓库根目录；`python scripts/toy_demo.py`；`python scripts/reproduce.py --check-only`；`python scripts/reproduce.py`。
- AI 工具及具体用途：读取自学材料、编排远程 workflow、生成汇总记录和周报；未改动科学算法、种子、阈值或输入。

## B. 输入与范围核对

| 项目 | 预期 | 实测 / 证据文件 | 是否一致 |
|---|---:|---|---|
| 测试视频 | 150 | donor 视频数 {audit['donor_video_count']} | 是 |
| mixed-label 审计目标 | 134 | {audit['evaluable_target_count']} | 是 |
| 全正常 / 全异常视频 | 10 / 6 | 保留在 150 视频 donor 池；不进入 134 目标均值 | 是 |
| 每个目标 donor | 8，均不等于目标 | {audit['donors_per_target']}；位移 `{audit['effective_donor_shifts']}` | 是 |
| 候选 / 哨兵数 | 3 / 4，全部报告 | 三候选和四哨兵均在下表 | 是 |
| 对齐规则 | 候选原始分数 `[1:]` 与标签严格等长 | 按 `evaluator_original.py` 原实现执行 | 是 |
| bootstrap | 视频单位；10000 次；种子 24082026 | 结果字段逐候选记录 `draws=10000`、`seed=24082026` | 是 |
| CNM 哨兵选择 | 每次抽样重新取四哨兵最大值 | `{result['simultaneous_inference']['max_sentinel_recomputed_within_each_draw']}` | 是 |
| 输入是否改动 | 否 | 输入哈希校验通过 | 是 |
| 是否新增训练或推理 | 否 | 仅使用保存预测 | 是 |

## C. 原实现重放结果

{mismatch_note}

| 候选 | 代码名称 | TIA 点估计 | TIA 95% 区间 | TIA 同时单侧下界 | CNM 点估计 | CNM 95% 区间 | CNM 同时单侧下界 | frame AUC | frame AP | 通过 |
|---|---|---:|---|---:|---:|---|---:|---:|---:|---|
{chr(10).join(candidate_lines)}

| 哨兵 | TIA 点估计 | 95% 区间 |
|---|---:|---|
{chr(10).join(sentinel_lines)}

| 候选 | 普通 frame AUC | 普通 frame AP | AP−异常帧比例 | CIL 检查 | 总判定 |
|---|---:|---:|---:|---|---|
{chr(10).join(ordinary_lines)}

- 五个分组结果、逐视频效应和完整布尔检查：已在 runner 临时 `REPLAY_RESULT.json` 中生成；本公开仓库只保留聚合记录。
- 字段比较方法：递归比较参考 JSON 与重算 JSON；float 使用 `rel_tol=1e-10`、`abs_tol=1e-12`。
- 输入/代码哈希：输入检查通过；环境哈希记录在 runner 临时输出中。

## D. 差异与异常记录

| 时间 / 位置 | 预期 | 实际观察 | 已检查的解释 | 尚不能排除的解释 | 下一步 / 是否需导师决定 |
|---|---|---|---|---|---|
| {date} / GitHub Actions | 玩具示例、输入校验、原实现重放完成 | 完成，比较结论：{comparison['matches_reference']} | 使用固定依赖 NumPy、原入口与原评估实现 | 未提交 runner 原始输出到公开仓库，需依赖 Actions 运行记录追溯完整日志 | 导师确认是否需要第二阶段 |

## E. 第二阶段方案

- 方案状态：未确认、未运行。
- 科学问题：四个哨兵的生成网格使用 `sentinels(N)` 还是 `sentinels(N+1)[1:]` 是否会改变现有审计结论？
- 唯一拟变因素：哨兵生成方式；目标和 donor 两侧一致采用条件 A 或 B。
- 固定项：150 视频 donor 池、134 mixed-label 目标、8 donor 映射、3 候选、4 哨兵、seed `24082026`、10000 次视频 bootstrap、门槛、区间算法和每次抽样重算最大哨兵。
- 停止条件：只比较 A/B，不扩展更多网格，不调参，不删候选。
- 如果不稳健：收窄主张，明确结果依赖网格约定。
- 如果稳健：仍不能声称模型理解内容或证明真实因果机制。
- 导师决定：不做追加 / 先核对协议来源 / 修改方案 / 同意固定范围运行 / 直接收窄结论。

## F. 结论与交接

- 已观察到的事实：首轮规定的玩具示例、输入校验与原实现重放已在 GitHub runner 完成。
- 由事实支持的最窄结论：本仓库保存输入在原实现下可追溯到参考结果；这是教学重放。
- 不能从中推出的结论：不能称为重新训练、独立实现、独立第三方复现或新的独立数据确认；也不能证明模型理解异常。
- 尚未解决的问题：学生仍需能独立解释 TIA、CNM、134/150 分母、donor 位移和普通检测指标。
- 能直接用于论文的文字草稿：已使用固定保存预测和原评估入口完成内部教学重放，结果在指定数值容差内与封存参考一致；该运行不构成新的独立确认。
- 完整输出、环境信息、日志与图表所在位置：GitHub Actions 本次运行日志；公开仓库 `reports/` 仅存聚合记录。
- 本次是否需要导师行动：需要决定第二阶段是否运行，或是否仅修正文档并收窄表述。
"""

    weekly = f"""# CIL 每周简报

**日期 / 姓名 / 本周可投入与实际投入：** {date} / 学生署名待补 / 可投入时间未提供；本次由 Codex 在 GitHub Actions 云端完成一次首轮重放。

**一句话进展：** 查清并执行了自学文档要求的开端流程：玩具示例、输入哈希校验、三候选原实现重放，并完成聚合实验记录。

| 科学问题 | 本周证据与文件 | 当前判断 | 仍不确定什么 |
|---|---|---|---|
| 自有曲线是否比其他视频 donor 更适合目标标签？ | GitHub Actions 重放；`reports/实验记录_{date}.md` | 三候选均按 TIA/CNM 完整报告，需按区间和检查解释 | 尚未展开单视频案例，需导师确认范围 |
| 简单时间规则能否解释优势？ | 四个 sentinel 的 TIA 和 CNM 扣减规则 | 已按预先规定哨兵族报告；不能说已穷尽所有非内容解释 | 哨兵网格敏感性是否要追加 |
| 这次运行的证据身份是什么？ | `REPLAY_RESULT.json` 的教学重放标记和本记录 | 是保存输入上的原实现重放，不是独立第三方复现 | 独立实现或新数据确认需另行设计 |

**已完成交付：** 在 GitHub Actions 中运行 `toy_demo.py`、`reproduce.py --check-only`、`reproduce.py`；提交聚合实验记录 `reports/实验记录_{date}.md` 和本周报 `reports/周报_{date}.md`。

**负结果与异常：** 未新增模型、未训练、未调参、未运行第二阶段。为避免把逐视频实验明细继续公开，未提交 runner 的 `outputs/` 原始目录。

**本周真正学会的一件事：** TIA 先在每个 mixed-label 目标视频上计算“自有曲线 AP 减 8 个 donor AP 均值”，再对 134 个目标平均；CNM 的最大哨兵需要在每次视频 bootstrap 抽样中重新计算。

**AI 帮了什么、我核验了什么：** AI 帮助读取任务书、搭建远程运行入口、整理报告；GitHub runner 实际执行固定代码和输入校验。科学解释、数据权限和第二阶段是否执行仍需学生与导师核验，不能只写“AI 已检查”。

**需要导师作出的决定：** 是否核对哨兵生成网格的协议来源；是否批准只在固定范围内运行第二阶段敏感性分析。

**下阶段最小计划：** 学生先独立回答自检题，并能口头解释三候选主表、134/150 分母、donor 索引位移和“教学重放”的证据边界；再由导师决定是否追加网格敏感性分析。
"""

    (REPORT_DIR / f"实验记录_{date}.md").write_text(record, encoding="utf-8")
    (REPORT_DIR / f"周报_{date}.md").write_text(weekly, encoding="utf-8")


if __name__ == "__main__":
    main()
