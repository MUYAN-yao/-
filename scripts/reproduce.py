"""Read-only scientific inputs; fresh, explicitly educational replay outputs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def verify_inputs():
    manifest = json.loads((ROOT / 'package_manifest.json').read_text(encoding='utf-8'))
    entries = manifest['files']
    required = {'scripts/evaluator_original.py', 'reference/RESULT.json',
                'inputs/SCIENTIFIC_CONTRACT_V1.json', 'inputs/ALIGNMENT_AMENDMENT_V1.json',
                'inputs/LABEL_VAULT_OPEN_RECEIPT.json',
                'inputs/PREDICTIONS_PRIMARY.zip', 'inputs/PREDICTIONS_REPLAY.zip'}
    if not required.issubset(entries):
        raise ValueError('Incomplete package manifest')
    checked = 0
    for name, info in entries.items():
        if not (name.startswith('inputs/') or name in required):
            continue
        p = (ROOT / name).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file():
            raise ValueError('Missing or unsafe input: ' + name)
        if p.stat().st_size != info['bytes'] or digest(p) != info['sha256']:
            raise ValueError('Input hash mismatch: ' + name)
        checked += 1
    receipt = json.loads((ROOT / 'inputs/LABEL_VAULT_OPEN_RECEIPT.json').read_text(encoding='utf-8'))
    labels_root = ROOT / 'inputs/labels'
    actual = {p.relative_to(labels_root).as_posix() for p in labels_root.rglob('*.npy')}
    expected = {row['path'] for row in receipt['labels']}
    if len(actual) != 150 or actual != expected:
        raise ValueError('Expected exactly the 150 historical label files')
    for row in receipt['labels']:
        if digest(labels_root / row['path']) != row['sha256']:
            raise ValueError('Historical label hash mismatch: ' + row['path'])
    return {'verified_input_files': checked, 'verified_historical_label_files': len(actual)}

def compare(expected, actual, path='', mismatches=None):
    if mismatches is None:
        mismatches = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            mismatches.append({'path': path, 'reason': 'keys/type differ'})
        else:
            for key in expected:
                compare(expected[key], actual[key], path + '/' + key, mismatches)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            mismatches.append({'path': path, 'reason': 'list length/type differs'})
        else:
            for i, (a, b) in enumerate(zip(expected, actual)):
                compare(a, b, f'{path}/{i}', mismatches)
    elif isinstance(expected, float):
        if not isinstance(actual, (float, int)) or not math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-12):
            mismatches.append({'path': path, 'expected': expected, 'actual': actual})
    elif type(expected) is not type(actual) or expected != actual:
        mismatches.append({'path': path, 'expected': expected, 'actual': actual})
    return mismatches

def write_json(path, data):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Verify every scientific input against the package and historical receipt; no metrics.')
    args = parser.parse_args()
    start = time.perf_counter()
    verified = verify_inputs()
    print('INPUT CHECK PASSED:', verified, flush=True)
    if args.check_only:
        return 0
    try:
        import numpy as np
        import evaluator_original as evaluator
    except ImportError as exc:
        raise RuntimeError('Install the dependency first: python -m pip install -r requirements.txt') from exc
    folder = ROOT / 'outputs' / ('replay_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    folder.mkdir(parents=True, exist_ok=False)
    print('Recomputing 3 candidates, 134 ownership targets, 10000 video-bootstrap draws. CPU only.', flush=True)
    result = evaluator.run(
        ROOT / 'inputs/SCIENTIFIC_CONTRACT_V1.json',
        ROOT / 'inputs/PREDICTIONS_PRIMARY.zip', ROOT / 'inputs/PREDICTIONS_REPLAY.zip',
        ROOT / 'inputs/labels', ROOT / 'inputs/LABEL_VAULT_OPEN_RECEIPT.json',
        ROOT / 'inputs/ALIGNMENT_AMENDMENT_V1.json')
    reference = json.loads((ROOT / 'reference/RESULT.json').read_text(encoding='utf-8'))
    differences = compare(reference, result)
    envelope = {
        'evidence_role': 'educational_replay_of_previously_consumed_historical_endpoint',
        'new_independent_confirmation': False,
        'independent_implementation': False,
        'note': 'Historical flags inside the recomputed object describe the source study, not this student replay.',
        'recomputed_historical_result': result,
    }
    write_json(folder / 'REPLAY_RESULT.json', envelope)
    write_json(folder / 'comparison.json', {'matches_reference': not differences,
        'numeric_tolerance': {'relative': 1e-10, 'absolute': 1e-12},
        'comparison_scope': 'all nested source-result keys, numbers, strings and decisions',
        'mismatches': differences, **verified})
    write_json(folder / 'environment.json', {'python': sys.version, 'numpy': np.__version__,
        'platform': platform.platform(), 'elapsed_seconds': time.perf_counter() - start,
        'utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': digest(Path(__file__)),
        'original_evaluator_sha256': digest(ROOT / 'scripts/evaluator_original.py')})
    fields = ['candidate', 'n', 'tia', 'tia_lower', 'tia_upper', 'cnm', 'cnm_lower', 'cnm_upper',
              'auc_all_150', 'ap_all_150', 'tia_simultaneous_lower', 'cnm_simultaneous_lower', 'qualified']
    rows = []
    for name in result['candidate_family']:
        candidate = result['candidate_results'][name]
        tia, cnm = candidate['target_identity_advantage'], candidate['content_necessity_margin_v2']
        metrics = result['ordinary_detection_metrics'][name]
        rows.append(dict(zip(fields, [name, result['dataset_audit']['evaluable_target_count'],
            tia['point'], tia['ci95_lower'], tia['ci95_upper'], cnm['point'], cnm['ci95_lower'], cnm['ci95_upper'],
            metrics['frame_auc'], metrics['frame_ap'], tia['simultaneous_one_sided_95_lower'],
            cnm['simultaneous_one_sided_95_lower'], candidate['qualified']])))
    with (folder / 'table_ii_iitb.csv').open('x', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    status = 'PASS' if not differences else 'DIFFERENCE DETECTED'
    report = '# CIL 教学重放结果\n\n' + status + '\n\n'
    report += '这是使用原实现对已使用端点的重算，不是第三方复现或新的独立确认。\n\n'
    report += '|候选|TIA|CNM|AUC（150视频）|AP（150视频）|\n|---|---:|---:|---:|---:|\n'
    for r in rows:
        report += f"|{r['candidate']}|{r['tia']:.6f}|{r['cnm']:.6f}|{r['auc_all_150']:.6f}|{r['ap_all_150']:.6f}|\n"
    report += '\n完整置信区间见 CSV，差异见 comparison.json。通过重放不等于已解释科研结论，请填写 templates 中的任务报告。\n'
    with (folder / 'REPORT.md').open('x', encoding='utf-8') as handle:
        handle.write(report)
    print(status + ': ' + folder.relative_to(ROOT).as_posix(), flush=True)
    return 0 if not differences else 2

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
