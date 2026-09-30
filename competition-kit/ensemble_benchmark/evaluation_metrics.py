"""Supplementary BSS Eval v4 diagnostics; existing official SDR is untouched.

Full-scene, fixed-filter SIR/SAR can require substantial RAM and CPU. Scene
results are cached with content hashes so interruption does not repeat scoring.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .scoring import SAMPLE_RATE, _path, _read_audio

METRIC_SETTINGS = dict(version=1, implementation='museval-0.4.1', window=44100,
                       hop=44100, filters_len=512, framewise_filters=False,
                       compute_permutation=False, bsseval_sources_version=False)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def prediction_provenance(manifest, predictions, checkpoint, release_id, settings):
    hashes = {}
    for scene in manifest['scenes']:
        for target in scene['target_classes']:
            relative = f"{scene['scene_id']}/{target}.wav"
            hashes[relative] = file_hash(_path(predictions, relative))
    return dict(version=1, checkpoint_sha256=file_hash(checkpoint), release_id=release_id,
                manifest_sha256=object_hash(manifest), prediction_sha256=hashes,
                inference=settings)


def validate_reuse(report, manifest, predictions, checkpoint, release_id,
                   provenance_path, trust_legacy=False, inference_settings=None):
    """Hash every prediction before reuse. Legacy origin cannot be proven retroactively."""
    if report.get('checkpoint_sha256') != file_hash(checkpoint):
        raise ValueError('Saved predictions belong to a different checkpoint; use a new run or rerun inference')
    if report.get('release_id') != release_id or report.get('diagnostic', False):
        raise ValueError('Saved prediction release/mode differs from full evaluation')
    expected = {(s['scene_id'], t) for s in manifest['scenes'] for t in s['target_classes']}
    actual = [(r['scene_id'], r['instrument']) for r in report['scene_scores']]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Saved report scene/target coverage differs from manifest')
    provenance_path = Path(provenance_path)
    previous = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    if previous is None and not trust_legacy:
        raise ValueError('Legacy predictions lack content hashes; explicitly set trust_legacy_predictions=True to adopt them')
    current = prediction_provenance(manifest, predictions, checkpoint, release_id,
                                    previous['inference'] if previous else {'legacy': 'unknown'})
    if previous:
        known = previous.get('inference', {})
        for key, value in (inference_settings or {}).items():
            if key in known and known[key] != value:
                raise ValueError(f'Saved inference setting differs: {key}; rerun inference explicitly')
        for key in ('checkpoint_sha256', 'release_id', 'manifest_sha256', 'prediction_sha256'):
            if current[key] != previous[key]:
                raise ValueError(f'Saved prediction provenance mismatch: {key}')
    else:
        # Check audio shape/format before adopting old outputs; no inferred provenance claims.
        import soundfile as sf
        for scene in manifest['scenes']:
            for target in scene['target_classes']:
                info = sf.info(_path(predictions, f"{scene['scene_id']}/{target}.wav"))
                if (info.frames != scene['num_samples'] or info.samplerate != SAMPLE_RATE
                        or info.channels != 1 or info.format != 'WAV' or info.subtype != 'FLOAT'):
                    raise ValueError('Legacy prediction format/length differs from manifest')
        current['legacy_adopted'] = True
        current['legacy_warning'] = 'User trusted origin; historical prediction hashes and inference settings unavailable'
        write_json(provenance_path, current)
    return previous or current


def score_scene_arrays(references, estimates, sample_rate=SAMPLE_RATE, filters_len=512):
    """Return SIR/SAR diagnostics for all sources jointly, never matching labels.

    Undefined/infinite frames are recorded rather than substituted by high scores.
    Entirely silent sources make joint BSS Eval undefined for the whole scene.
    """
    references = np.asarray(references, dtype=np.float64)
    estimates = np.asarray(estimates, dtype=np.float64)
    if references.ndim != 2 or references.shape != estimates.shape or not references.size:
        raise ValueError('Expected matching nonempty source x sample arrays')
    if not np.isfinite(references).all() or not np.isfinite(estimates).all():
        raise ValueError('Nonfinite reference/estimate audio')
    padding = (-references.shape[1]) % sample_rate
    references = np.pad(references, ((0, 0), (0, padding)))
    estimates = np.pad(estimates, ((0, 0), (0, padding)))
    active = np.mean(references.reshape(len(references), -1, sample_rate)**2, axis=2) > 1e-12
    frames = active.shape[1]
    reason = None
    if np.any(np.all(references == 0, axis=1)) or np.any(np.all(estimates == 0, axis=1)):
        reason = 'at_least_one_entire_source_is_silent'
        sir = sar = np.full(active.shape, np.nan)
    else:
        from museval.metrics import bss_eval
        _, _, sir, sar, _ = bss_eval(references[:, :, None], estimates[:, :, None],
            window=sample_rate, hop=sample_rate, filters_len=filters_len,
            framewise_filters=False, compute_permutation=False, bsseval_sources_version=False)
    results = []
    for index in range(len(references)):
        row = {'active_frames': int(active[index].sum()), 'total_frames': frames}
        for metric, values in (('sir', sir[index]), ('sar', sar[index])):
            eligible = values[active[index]]
            finite = eligible[np.isfinite(eligible)]
            undefined = int(np.isnan(eligible).sum())
            positive = int(np.isposinf(eligible).sum())
            negative = int(np.isneginf(eligible).sum())
            complete = bool(eligible.size and finite.size == eligible.size)
            row[metric] = float(np.median(finite)) if finite.size else None
            row[metric + '_diagnostics'] = dict(valid_frames=int(finite.size),
                undefined_frames=undefined, positive_infinite_frames=positive,
                negative_infinite_frames=negative, complete=complete,
                finite_frame_median=float(np.median(finite)) if finite.size else None,
                reason=reason)
        results.append(row)
    return results


def summarize(rows, metric):
    """SDR hierarchy; partial medians remain diagnostics, never complete scores."""
    parents = defaultdict(list)
    invalid = set()
    for row in rows:
        dataset, instrument = row['dataset'], row['instrument']
        if row.get(metric) is None or not row.get(metric + '_diagnostics', {}).get('complete', True):
            invalid.add((dataset, instrument))
        if row.get(metric) is not None:
            parents[(dataset, instrument, row['work_group'], str(row['parent_id']))].append(row[metric])
    works = defaultdict(list)
    for (dataset, instrument, work, parent), values in parents.items():
        works[(dataset, instrument, work)].append(float(np.median(values)))
    instruments = defaultdict(list)
    for (dataset, instrument, work), values in works.items():
        instruments[(dataset, instrument)].append(float(np.median(values)))
    datasets = {}
    for dataset in sorted({r['dataset'] for r in rows}):
        targets = sorted({r['instrument'] for r in rows if r['dataset'] == dataset})
        scores = {name: (float(np.median(instruments[(dataset, name)]))
                  if instruments[(dataset, name)] else None) for name in targets}
        valid = [v for v in scores.values() if v is not None]
        complete = len(valid) == len(scores) and bool(valid) and not any((dataset, t) in invalid for t in targets)
        datasets[dataset] = dict(instruments=scores, complete=complete,
            score=float(np.mean(valid)) if complete else None,
            valid_instrument_mean=float(np.mean(valid)) if valid else None,
            invalid_instruments=[name for name in targets if (dataset, name) in invalid],
            instrument_coverage={name: {
                'active_frames': sum(r.get('active_frames', 0) for r in rows if r['dataset']==dataset and r['instrument']==name),
                'valid_frames': sum(r.get(metric+'_diagnostics', {}).get('valid_frames', 0) for r in rows if r['dataset']==dataset and r['instrument']==name),
                'complete': (dataset, name) not in invalid} for name in targets})
    complete = bool(datasets) and all(d['complete'] for d in datasets.values())
    return dict(datasets=datasets, complete=complete,
                overall=float(np.mean([d['score'] for d in datasets.values()])) if complete else None)


def ensemble_groups(manifest, rows):
    scenes = {s['scene_id']: s for s in manifest['scenes']}
    groups = defaultdict(list)
    for row in rows:
        scene = scenes[row['scene_id']]
        kind = 'part_count' if 'part_count' in scene else 'target_class_count'
        count = scene.get('part_count', len(scene['target_classes']))
        groups[f'{kind}={count}'].append(row)
    return {key: {metric: summarize(values, metric) for metric in ('sdr', 'sir', 'sar')}
            for key, values in groups.items()}


def add_supplementary(report, manifest, references_root, predictions, cache_dir):
    """Compute or reuse per-scene supplementary scores without model inference."""
    cache_dir = Path(cache_dir)
    rows = []
    for scene in tqdm(manifest['scenes'], desc='SIR/SAR scoring (full scene)', unit='scene'):
        targets = scene['target_classes']
        if set(targets) != set(scene['references']):
            raise ValueError('Joint SIR/SAR requires all scene reference sources')
        refs = [_path(references_root, scene['references'][t]) for t in targets]
        preds = [_path(predictions, f"{scene['scene_id']}/{t}.wav") for t in targets]
        key = object_hash(dict(settings=METRIC_SETTINGS, scene=scene,
                              references=[file_hash(p) for p in refs], predictions=[file_hash(p) for p in preds]))
        cache = cache_dir / (key + '.json')
        if cache.exists():
            values = json.loads(cache.read_text())
        else:
            estimated_gib = scene['num_samples'] * len(targets) * 8 * 2 / 2**30
            tqdm.write(f"{scene['scene_id']}: reference+estimate arrays >= {estimated_gib:.2f} GiB; BSS Eval FFT/filter workspace requires additional RAM")
            reference = np.stack([_read_audio(p, reference=True) for p in refs])
            estimate = np.stack([_read_audio(p) for p in preds])
            if reference.shape[1] != scene['num_samples']:
                raise ValueError('Reference length differs from manifest')
            values = score_scene_arrays(reference, estimate)
            write_json(cache, values)
            del reference, estimate
        for target, value in zip(targets, values):
            row = {k: scene[k] for k in ('scene_id', 'dataset', 'parent_id', 'work_group')}
            rows.append(dict(row, instrument=target, **value))
    combined = {(r['scene_id'], r['instrument']): dict(r) for r in report['scene_scores']}
    for row in rows:
        combined[(row['scene_id'], row['instrument'])].update(row)
    result = dict(report)
    result['supplementary'] = dict(settings=METRIC_SETTINGS,
        undefined_policy='Instrument values use finite active-frame medians; incomplete coverage is diagnostic only. Overall and dataset score require all active frames finite. Museval masks a frame if any reference or estimate source is silent.',
        sir=summarize(rows, 'sir'), sar=summarize(rows, 'sar'), scene_scores=rows)
    result['ensemble_groups'] = ensemble_groups(manifest, list(combined.values()))
    return result
