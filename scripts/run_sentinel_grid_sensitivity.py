"""Run the pre-specified A/B sentinel-grid sensitivity analysis."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import evaluator_original as evaluator
from reproduce import verify_inputs


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
CANDIDATES = tuple(evaluator.CANDIDATES)
SENTINELS = tuple(evaluator.SENTINELS)


def aligned_grid_sentinels(frame_count: int):
    generated = evaluator_sentinels(frame_count + 1)
    aligned = {name: values[1:] for name, values in generated.items()}
    if any(len(values) != frame_count for values in aligned.values()):
        raise AssertionError("sentinel grid alignment failed")
    return aligned


def run_condition(condition: str):
    if condition == "A_original_grid":
        evaluator.sentinels = evaluator_sentinels
    elif condition == "B_aligned_grid":
        evaluator.sentinels = aligned_grid_sentinels
    else:
        raise ValueError(f"unknown condition: {condition}")
    return evaluator.run(
        ROOT / "inputs/SCIENTIFIC_CONTRACT_V1.json",
        ROOT / "inputs/PREDICTIONS_PRIMARY.zip",
        ROOT / "inputs/PREDICTIONS_REPLAY.zip",
        ROOT / "inputs/labels",
        ROOT / "inputs/LABEL_VAULT_OPEN_RECEIPT.json",
        ROOT / "inputs/ALIGNMENT_AMENDMENT_V1.json",
    )


def candidate_summary(result, name: str):
    item = result["candidate_results"][name]
    tia = item["target_identity_advantage"]
    cnm = item["content_necessity_margin_v2"]
    return {
        "qualified": item["qualified"],
        "checks": item["checks"],
        "tia": tia,
        "cnm": cnm,
    }


def make_report(a, b, verified):
    rows = []
    status_changed = False
    for name in CANDIDATES:
        left = candidate_summary(a, name)
        right = candidate_summary(b, name)
        changed = left["qualified"] != right["qualified"] or left["checks"] != right["checks"]
        status_changed = status_changed or changed
        rows.append(
            "| {name} | {atia:.6f} | {acnm:.6f} | {btia:.6f} | {bcnm:.6f} | {delta:.6f} | {aq} | {bq} | {changed} |".format(
                name=name,
                atia=left["tia"]["point"],
                acnm=left["cnm"]["point"],
                btia=right["tia"]["point"],
                bcnm=right["cnm"]["point"],
                delta=right["cnm"]["point"] - left["cnm"]["point"],
                aq=left["qualified"],
                bq=right["qualified"],
                changed="是" if changed else "否",
            )
        )
    decision_changed = a["paper_promotion_gate_passed"] != b["paper_promotion_gate_passed"]
    robust = not status_changed and not decision_changed
    sentinel_rows = []
    for name in SENTINELS:
        sentinel_rows.append(
            "| {name} | {a:.6f} | {b:.6f} | {delta:.6f} |".format(
                name=name,
                a=a["sentinel_results"][name]["point"],
                b=b["sentinel_results"][name]["point"],
                delta=b["sentinel_results"][name]["point"] - a["sentinel_results"][name]["point"],
            )
        )
    conclusion = (
        "三候选的全部检查和论文推广判定在两个条件下均未改变；首轮结论对这一项网格变化稳健。"
        if robust
        else "至少一个候选检查或论文推广判定改变；首轮结论依赖该网格约定，应收窄表述并报告全部 A/B 结果。"
    )
    report = f"""# 第二阶段哨兵网格敏感性分析

**状态：** 已执行的事后敏感性分析，不是新的独立确认。

## 固定方案

- A：四个哨兵使用 `sentinels(N)`；B：使用 `sentinels(N+1)[1:]`。
- 候选均保持原实现的 `score[1:]`；标签、150 个 donor 池、134 个目标、8 donor 位移、10,000 次视频级 bootstrap、种子 `24082026`、门槛和候选族均不变。
- CNM 在每次 bootstrap 抽样中重新取四哨兵的最大值。
- 输入校验：{verified}。

## 三候选结果

| 候选 | A TIA | A CNM | B TIA | B CNM | B−A CNM | A 通过 | B 通过 | 检查改变 |
|---|---:|---:|---:|---:|---:|---|---|---|
{chr(10).join(rows)}

## 哨兵 TIA

| 哨兵 | A | B | B−A |
|---|---:|---:|---:|
{chr(10).join(sentinel_rows)}

## 判读

{conclusion}

完整 A/B 结果、区间和布尔检查仅保留在本次 Actions runner 的 `outputs/` 中；公开仓库仅保存本聚合报告。结果不证明模型理解异常，也不替代独立实现或新数据确认。
"""
    return report, robust


def main():
    global evaluator_sentinels
    verified = verify_inputs()
    evaluator_sentinels = evaluator.sentinels
    try:
        baseline = run_condition("A_original_grid")
        aligned = run_condition("B_aligned_grid")
    finally:
        evaluator.sentinels = evaluator_sentinels

    report, robust = make_report(baseline, aligned, verified)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = ROOT / "outputs" / f"sentinel_grid_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "SENSITIVITY_RESULT.json").write_text(
        json.dumps(
            {
                "analysis": "post_hoc_sentinel_grid_sensitivity",
                "condition_a": "sentinels(N)",
                "condition_b": "sentinels(N+1)[1:]",
                "fixed_factors": "candidate score[1:], labels, donors, bootstrap, seed and gates",
                "verified_inputs": verified,
                "conclusion_robust_for_this_variation": robust,
                "condition_a_result": baseline,
                "condition_b_result": aligned,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    report_path = ROOT / "reports" / "第二阶段_哨兵网格敏感性分析_2026-09-16.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"SENSITIVITY COMPLETE: robust={robust}")


if __name__ == "__main__":
    main()
