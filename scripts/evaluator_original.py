"""One-shot evaluation of the frozen IITB Corridor prospective CIL family."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCOPE = "iitb_corridor_prospective_cil_v1"
CANDIDATES = (
    "resnet18_normal_diagonal",
    "resnet18_temporal_novelty",
    "pixel_frame_difference",
)
SENTINELS = (
    "sentinel_center_prior",
    "sentinel_duration_center",
    "sentinel_linear_progress",
    "sentinel_sampling_grid",
)


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def rank_percentile(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("rank percentile requires at least two finite values")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return (ranks - 0.5) / len(values)


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("AUC inputs must be aligned finite vectors")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not set(np.unique(labels)).issubset({0, 1}) or positives == 0 or negatives == 0:
        raise ValueError("AUC requires both binary classes")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("AP inputs must be aligned finite vectors")
    if not set(np.unique(labels)).issubset({0, 1}) or labels.sum() == 0:
        raise ValueError("AP requires binary labels with positives")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(scores) - 1]
    true_positive = np.cumsum(sorted_labels)[ends].astype(np.float64)
    precision = true_positive / (ends.astype(np.float64) + 1.0)
    recall = true_positive / float(labels.sum())
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def sentinels(frame_count: int) -> dict[str, np.ndarray]:
    if frame_count < 2:
        raise ValueError("sentinels require at least two frames")
    index = np.arange(frame_count, dtype=np.float64)
    progress = index / float(frame_count - 1)
    width = float(np.clip(8.0 / frame_count, 0.12, 0.45))
    return {
        "sentinel_center_prior": np.exp(-0.5 * ((progress - 0.5) / 0.22) ** 2),
        "sentinel_duration_center": np.exp(-0.5 * ((progress - 0.5) / width) ** 2),
        "sentinel_linear_progress": progress,
        "sentinel_sampling_grid": np.sin(np.pi * (index + 0.5) / 4.0) ** 2,
    }


def hash_order(values: Sequence[str], namespace: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{SCOPE}|{namespace}|{value}".encode("utf-8")
        ).hexdigest(),
    )


def effective_shifts(raw_shifts: Sequence[int], count: int) -> tuple[int, ...]:
    shifts: list[int] = []
    for raw in raw_shifts:
        value = int(raw) % count
        if not value or value in shifts:
            raise ValueError("donor shifts collapse or self-map")
        shifts.append(value)
    return tuple(shifts)


def progress_match(curve: np.ndarray, target_length: int) -> np.ndarray:
    return np.interp(
        np.linspace(0.0, 1.0, target_length),
        np.linspace(0.0, 1.0, len(curve)),
        curve,
    )


def interval(point: float, draws: np.ndarray, draw_count: int, seed: int) -> dict[str, Any]:
    return {
        "point": float(point),
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
        "draws": draw_count,
        "seed": seed,
        "unit": "video",
    }


def validate_contract(contract: Mapping[str, Any]) -> None:
    candidate_names = tuple(item["name"] for item in contract.get("candidate_family", ()))
    exact = {
        "scope": SCOPE,
        "locked": True,
        "expected_train_videos": 208,
        "expected_test_videos": 150,
        "minimum_donor_videos": 100,
        "minimum_evaluable_targets": 60,
        "donor_raw_shifts": [7, 19, 31, 43, 59, 73, 91, 109],
        "bootstrap_draws": 10000,
        "bootstrap_seed": 24082026,
        "strata": 5,
        "identity_floor": 0.02,
        "margin_floor": 0.015,
        "report_all_candidates": True,
        "prediction_replay": "byte_identical_required",
    }
    for key, expected in exact.items():
        if contract.get(key) != expected:
            raise ValueError(f"contract drift at {key}")
    if candidate_names != CANDIDATES or tuple(contract.get("sentinels", ())) != SENTINELS:
        raise ValueError("candidate or sentinel family drift")
    utility = contract.get("ordinary_utility", {})
    if utility.get("paper_promotion_auc_floor") != 0.60 or utility.get(
        "paper_promotion_ap_above_prevalence"
    ) != 0.05:
        raise ValueError("paper-promotion utility gate drift")


def load_predictions(path: Path, contract_sha256: str) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    curves = {candidate: {} for candidate in CANDIDATES}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "manifest.json" not in names:
            raise ValueError("prediction archive structure invalid")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("labels_read") is not False:
            raise ValueError("prediction was not generated blind")
        if manifest.get("contract_sha256") != contract_sha256:
            raise ValueError("prediction contract hash mismatch")
        if tuple(manifest.get("candidate_order", ())) != CANDIDATES:
            raise ValueError("prediction family mismatch")
        if len(manifest.get("videos", ())) != 150:
            raise ValueError("prediction video count mismatch")
        seen: set[str] = set()
        for record in manifest["videos"]:
            video_id = str(record["video_id"])
            key = video_id.casefold()
            if key in seen:
                raise ValueError("duplicate prediction video ID")
            seen.add(key)
            frame_count = int(record["frame_count"])
            for candidate in CANDIDATES:
                score_record = record["scores"][candidate]
                entry = str(score_record["entry"])
                body = archive.read(entry)
                if hashlib.sha256(body).hexdigest() != score_record["sha256"]:
                    raise ValueError(f"prediction entry hash mismatch: {entry}")
                array = np.asarray(np.load(io.BytesIO(body), allow_pickle=False), dtype=np.float64)
                if array.shape != (frame_count,) or not np.isfinite(array).all():
                    raise ValueError(f"invalid prediction curve: {video_id}/{candidate}")
                curves[candidate][video_id] = array
    return curves, manifest


def load_labels(root: Path, video_ids: Sequence[str]) -> dict[str, np.ndarray]:
    files = sorted(item for item in root.rglob("*.npy") if item.is_file())
    index: dict[str, Path] = {}
    for path in files:
        key = path.stem.casefold()
        if key in index:
            raise ValueError(f"duplicate label basename: {path.stem}")
        index[key] = path
    labels: dict[str, np.ndarray] = {}
    for video_id in video_ids:
        path = index.get(video_id.casefold())
        if path is None:
            raise ValueError(f"label missing for video {video_id}")
        array = np.asarray(np.load(path, allow_pickle=False))
        if array.ndim != 1 or not set(np.unique(array).tolist()).issubset({0, 1}):
            raise ValueError(f"invalid binary frame labels for {video_id}: shape={array.shape}")
        labels[video_id] = array.astype(np.int64, copy=False)
    if len(files) != len(video_ids):
        raise ValueError("label vault has unmatched files")
    return labels


def apply_frozen_transition_alignment(
    curves: Mapping[str, Mapping[str, np.ndarray]],
    labels: Mapping[str, np.ndarray],
    amendment: Mapping[str, Any],
    prediction_sha256: str,
    contract_sha256: str,
    label_manifest_sha256: str,
) -> dict[str, dict[str, np.ndarray]]:
    if (
        amendment.get("scope") != "iitb_corridor_outcome_blind_alignment_amendment_v1"
        or amendment.get("locked") is not True
        or amendment.get("discovered_before_any_metric_computation") is not True
        or amendment.get("failed_attempt_consumed_metrics") is not False
        or amendment.get("frozen_rule")
        != "drop_score_frame_zero_then_require_exact_label_length"
        or amendment.get("prediction_sha256") != prediction_sha256
        or amendment.get("scientific_contract_sha256") != contract_sha256
        or amendment.get("label_manifest_sha256") != label_manifest_sha256
        or amendment.get("metrics_computed_at_lock") != 0
    ):
        raise ValueError("outcome-blind alignment amendment is invalid")
    audit = amendment.get("structural_audit", {})
    if (
        audit.get("test_videos") != 150
        or audit.get("videos_with_label_length_equal_decoded_frames_minus_one") != 150
        or audit.get("other_length_relations") != 0
    ):
        raise ValueError("alignment structural audit is not unanimous")
    aligned: dict[str, dict[str, np.ndarray]] = {name: {} for name in CANDIDATES}
    for name in CANDIDATES:
        for video_id, label in labels.items():
            curve = np.asarray(curves[name][video_id], dtype=np.float64)
            if len(curve) != len(label) + 1:
                raise ValueError(f"non-unanimous one-frame alignment: {video_id}/{name}")
            aligned[name][video_id] = curve[1:]
            if len(aligned[name][video_id]) != len(label):
                raise AssertionError("transition alignment failed")
    return aligned


def per_video_effects(
    all_curves: Mapping[str, Mapping[str, np.ndarray]],
    labels: Mapping[str, np.ndarray],
    donor_map: Mapping[str, Sequence[str]],
    evaluable: Sequence[str],
) -> dict[str, dict[str, float]]:
    ranked = {
        name: {video_id: rank_percentile(curve) for video_id, curve in rows.items()}
        for name, rows in all_curves.items()
    }
    effects = {name: {} for name in all_curves}
    for name, rows in ranked.items():
        for video_id in evaluable:
            actual = average_precision(labels[video_id], rows[video_id])
            donors = [
                average_precision(
                    labels[video_id], progress_match(rows[donor], len(labels[video_id]))
                )
                for donor in donor_map[video_id]
            ]
            effects[name][video_id] = actual - float(np.mean(donors))
    return effects


def evaluate(
    candidate_curves: Mapping[str, Mapping[str, np.ndarray]],
    labels: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    video_ids = sorted(labels)
    if len(video_ids) != int(contract["expected_test_videos"]):
        raise ValueError("evaluated video count mismatch")
    for video_id in video_ids:
        for candidate in CANDIDATES:
            if len(candidate_curves[candidate][video_id]) != len(labels[video_id]):
                raise ValueError(f"frame alignment mismatch: {video_id}/{candidate}")
    evaluable = sorted(
        video_id for video_id in video_ids if set(labels[video_id].tolist()) == {0, 1}
    )
    if len(video_ids) < int(contract["minimum_donor_videos"]):
        raise ValueError("donor-video floor not met")
    if len(evaluable) < int(contract["minimum_evaluable_targets"]):
        raise ValueError("evaluable-target floor not met")
    all_curves: dict[str, dict[str, np.ndarray]] = {
        name: dict(rows) for name, rows in candidate_curves.items()
    }
    for sentinel in SENTINELS:
        all_curves[sentinel] = {
            video_id: sentinels(len(labels[video_id]))[sentinel] for video_id in video_ids
        }
    donor_order = hash_order(video_ids, "donor")
    shifts = effective_shifts(contract["donor_raw_shifts"], len(donor_order))
    donor_map = {
        video_id: [donor_order[(index + shift) % len(donor_order)] for shift in shifts]
        for index, video_id in enumerate(donor_order)
    }
    effects = per_video_effects(all_curves, labels, donor_map, evaluable)
    arrays = {
        name: np.asarray([rows[video_id] for video_id in evaluable], dtype=np.float64)
        for name, rows in effects.items()
    }
    draw_count = int(contract["bootstrap_draws"])
    seed = int(contract["bootstrap_seed"])
    generator = np.random.default_rng(seed)
    draw_indices = generator.integers(0, len(evaluable), size=(draw_count, len(evaluable)))
    points = {name: float(values.mean()) for name, values in arrays.items()}
    draws = {name: values[draw_indices].mean(axis=1) for name, values in arrays.items()}
    sentinel_point_max = max(points[name] for name in SENTINELS)
    sentinel_draw_max = np.max(np.vstack([draws[name] for name in SENTINELS]), axis=0)
    margin_points = {name: points[name] - sentinel_point_max for name in CANDIDATES}
    margin_draws = {name: draws[name] - sentinel_draw_max for name in CANDIDATES}
    identity_deviation = np.vstack([points[name] - draws[name] for name in CANDIDATES])
    margin_deviation = np.vstack(
        [margin_points[name] - margin_draws[name] for name in CANDIDATES]
    )
    identity_critical = float(np.quantile(np.max(identity_deviation, axis=0), 0.95))
    margin_critical = float(np.quantile(np.max(margin_deviation, axis=0), 0.95))
    strata_count = int(contract["strata"])
    strata_map = {
        video_id: index % strata_count
        for index, video_id in enumerate(hash_order(evaluable, "strata"))
    }
    candidate_results: dict[str, Any] = {}
    for name in CANDIDATES:
        identity_strata: list[float] = []
        margin_strata: list[float] = []
        for stratum in range(strata_count):
            mask = np.asarray([strata_map[video_id] == stratum for video_id in evaluable])
            candidate_mean = float(arrays[name][mask].mean())
            identity_strata.append(candidate_mean)
            margin_strata.append(
                candidate_mean
                - max(float(arrays[sentinel][mask].mean()) for sentinel in SENTINELS)
            )
        identity = interval(points[name], draws[name], draw_count, seed)
        identity["simultaneous_one_sided_95_lower"] = points[name] - identity_critical
        margin = interval(margin_points[name], margin_draws[name], draw_count, seed)
        margin["simultaneous_one_sided_95_lower"] = margin_points[name] - margin_critical
        checks = {
            "identity_point_at_least_floor": identity["point"] >= float(contract["identity_floor"]),
            "identity_ci95_lower_positive": identity["ci95_lower"] > 0.0,
            "identity_simultaneous_lower_positive": identity["simultaneous_one_sided_95_lower"] > 0.0,
            "margin_point_at_least_floor": margin["point"] >= float(contract["margin_floor"]),
            "margin_ci95_lower_positive": margin["ci95_lower"] > 0.0,
            "margin_simultaneous_lower_positive": margin["simultaneous_one_sided_95_lower"] > 0.0,
            "identity_nonnegative_strata_at_least_4_of_5": sum(x >= 0 for x in identity_strata) >= 4,
            "margin_nonnegative_strata_at_least_4_of_5": sum(x >= 0 for x in margin_strata) >= 4,
        }
        candidate_results[name] = {
            "qualified": all(checks.values()),
            "checks": checks,
            "target_identity_advantage": identity,
            "content_necessity_margin_v2": margin,
            "identity_strata": identity_strata,
            "margin_strata": margin_strata,
            "per_video_effects": {
                video_id: float(effects[name][video_id]) for video_id in evaluable
            },
        }
    global_labels = np.concatenate([labels[video_id] for video_id in video_ids])
    prevalence = float(global_labels.mean())
    ordinary: dict[str, Any] = {}
    for name in CANDIDATES:
        global_scores = np.concatenate([candidate_curves[name][video_id] for video_id in video_ids])
        auc = roc_auc(global_labels, global_scores)
        ap = average_precision(global_labels, global_scores)
        ordinary[name] = {
            "frame_auc": auc,
            "frame_ap": ap,
            "frame_prevalence": prevalence,
            "ap_above_prevalence": ap - prevalence,
            "auc_at_least_0_60": auc >= 0.60,
            "ap_at_least_prevalence_plus_0_05": ap >= prevalence + 0.05,
        }
    qualified = [name for name in CANDIDATES if candidate_results[name]["qualified"]]
    promoted = [
        name
        for name in qualified
        if ordinary[name]["auc_at_least_0_60"]
        and ordinary[name]["ap_at_least_prevalence_plus_0_05"]
    ]
    decision = (
        "paper_promotion_gate_passed"
        if promoted
        else "cil_pass_low_ordinary_utility"
        if qualified
        else "scientific_negative"
    )
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "decision": decision,
        "cil_gate_passed": bool(qualified),
        "paper_promotion_gate_passed": bool(promoted),
        "candidate_family": list(CANDIDATES),
        "complete_family_reported": True,
        "qualified_candidates": qualified,
        "paper_promoted_candidates": promoted,
        "candidate_results": candidate_results,
        "sentinel_results": {
            name: interval(points[name], draws[name], draw_count, seed) for name in SENTINELS
        },
        "ordinary_detection_metrics": ordinary,
        "dataset_audit": {
            "donor_video_count": len(video_ids),
            "evaluable_target_count": len(evaluable),
            "analyzed_frame_count": len(global_labels),
            "positive_frame_count": int(global_labels.sum()),
            "frame_prevalence": prevalence,
            "effective_donor_shifts": list(shifts),
            "donors_per_target": len(shifts),
            "evaluable_video_ids_sha256": hashlib.sha256(
                ("\n".join(evaluable) + "\n").encode("utf-8")
            ).hexdigest(),
        },
        "simultaneous_inference": {
            "method": "paired_video_bootstrap_max_deviation_one_sided_95",
            "candidate_count": len(CANDIDATES),
            "identity_critical_value": identity_critical,
            "margin_critical_value": margin_critical,
            "max_sentinel_recomputed_within_each_draw": True,
        },
        "claim_boundary": (
            "Independent real-surveillance confirmation is permitted only when "
            "paper_promotion_gate_passed is true; all candidates remain reportable."
        ),
    }


def run(
    contract_path: Path,
    primary: Path,
    replay: Path,
    labels_root: Path,
    label_receipt_path: Path,
    alignment_path: Path,
) -> dict[str, Any]:
    contract_body = contract_path.read_bytes()
    contract = json.loads(contract_body)
    validate_contract(contract)
    contract_sha256 = hashlib.sha256(contract_body).hexdigest()
    primary_sha256 = sha256_file(primary)
    replay_sha256 = sha256_file(replay)
    if primary_sha256 != replay_sha256 or primary.read_bytes() != replay.read_bytes():
        raise ValueError("primary and replay predictions are not byte-identical")
    receipt = json.loads(label_receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "labels_opened_once"
        or receipt.get("predictions_sealed_before_open") is not True
        or receipt.get("prediction_sha256") != primary_sha256
        or receipt.get("contract_sha256") != contract_sha256
    ):
        raise ValueError("label-vault opening receipt is invalid")
    curves, manifest = load_predictions(primary, contract_sha256)
    video_ids = [str(item["video_id"]) for item in manifest["videos"]]
    labels = load_labels(labels_root, video_ids)
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    curves = apply_frozen_transition_alignment(
        curves,
        labels,
        alignment,
        primary_sha256,
        contract_sha256,
        receipt["label_manifest_sha256"],
    )
    result = evaluate(curves, labels, contract)
    result["evidence_class"] = contract["evidence_class_if_all_gates_pass"]
    result["labels_opened_once"] = True
    result["prediction_primary_replay_byte_identical"] = True
    result["input_hashes"] = {
        "contract_sha256": contract_sha256,
        "prediction_sha256": primary_sha256,
        "label_open_receipt_sha256": sha256_file(label_receipt_path),
        "alignment_amendment_sha256": sha256_file(alignment_path),
        "label_manifest_sha256": receipt["label_manifest_sha256"],
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--label-receipt", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seal_path = args.output.with_suffix(args.output.suffix + ".seal.json")
    if args.output.exists() or seal_path.exists():
        raise FileExistsError("refusing to overwrite one-shot result or seal")
    result = run(
        args.contract.resolve(),
        args.primary.resolve(),
        args.replay.resolve(),
        args.labels_root.resolve(),
        args.label_receipt.resolve(),
        args.alignment.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.write_text(body, encoding="utf-8")
    seal_path.write_text(
        json.dumps(
            {
                "scope": SCOPE,
                "result": args.output.name,
                "result_sha256": sha256_file(args.output),
                "decision": result["decision"],
                "cil_gate_passed": result["cil_gate_passed"],
                "paper_promotion_gate_passed": result["paper_promotion_gate_passed"],
                "candidate_family": result["candidate_family"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "qualified_candidates": result["qualified_candidates"],
                "paper_promoted_candidates": result["paper_promoted_candidates"],
                "result_sha256": sha256_file(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
