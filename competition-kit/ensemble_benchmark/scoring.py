"""Competition SDR scorer (museval 0.4.1, BSS Eval v4 images).

References are fixed labels, never permutation matched. Each target is evaluated
separately: images SDR is independent of the other reference sources, while
upstream multi-source silence checks otherwise discard valid target frames.
Full recordings are evaluated with fixed global filters, not framewise filters.
The file scorer streams one-second blocks. For the images SDR only, projection
errors cancel algebraically; no global FFT or full-recording buffer is needed.
The array API retains museval as a small-signal reference implementation.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

SAMPLE_RATE = 44100
CLASSES = ('Violin', 'Viola', 'Cello', 'Bass', 'Flute', 'Clarinet', 'Oboe',
           'Bassoon', 'Horn', 'Trumpet', 'Trombone', 'Tuba', 'Harp', 'Timpani',
           'untunedpercussion')
EXPECTED_CLASSES = {'SynthSOD': CLASSES, 'URMP': CLASSES[:12]}


class SubmissionError(ValueError):
    """Malformed submitted audio or missing required files."""


class ScoringError(ValueError):
    """Invalid reference or unresolved numerical scoring failure."""


def score_target(reference, estimate, sample_rate=SAMPLE_RATE, min_active_frames=5):
    """Return clipped frame SDR and median for a single mono target.

    NaNs in reference-inactive frames are omitted; all other NaNs are errors.
    Exactly muted active estimates receive -60, even when museval emits NaN.
    The final fractional second is zero padded before both masking and scoring.
    """
    if sample_rate != SAMPLE_RATE:
        raise SubmissionError('Expected 44100 Hz')
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.ndim != 1 or reference.size == 0 or not np.isfinite(reference).all():
        raise ScoringError('Reference must be finite nonempty mono audio')
    if estimate.shape != reference.shape or not np.isfinite(estimate).all():
        raise SubmissionError('Estimate must match mono reference shape and be finite')
    padding = (-reference.size) % sample_rate
    reference = np.pad(reference, (0, padding))
    estimate = np.pad(estimate, (0, padding))
    ref_frames = reference.reshape(-1, sample_rate)
    est_frames = estimate.reshape(-1, sample_rate)
    active = np.mean(ref_frames ** 2, axis=1) > 1e-12
    if np.count_nonzero(active) < min_active_frames:
        raise ScoringError(f'Reference has fewer than {min_active_frames} active frames')
    muted = np.all(est_frames == 0, axis=1)
    values = np.full(active.size, np.nan)
    if np.any(active & ~muted):
        from museval.metrics import bss_eval
        # Explicit 3D arrays: source, sample, channel. Never run on one-second
        # chunks: that would refit filters and change BSS Eval v4 into v3.
        values = bss_eval(reference[None, :, None], estimate[None, :, None],
                          window=sample_rate, hop=sample_rate,
                          compute_permutation=False, filters_len=512,
                          framewise_filters=False,
                          bsseval_sources_version=False)[0][0]
        if values.shape != active.shape:
            raise ScoringError('Metric frame count differs from reference mask')
    values[active & muted] = -60.0
    if np.isnan(values[active]).any():
        raise ScoringError('NaN SDR on active reference; investigate metric/input')
    values = np.clip(values, -60.0, 60.0)
    return {'sdr': float(np.median(values[active])),
            'active_frames': int(active.sum()), 'total_frames': int(active.size),
            'frame_sdr': [float(v) if a else None for v, a in zip(values, active)]}


def aggregate_scores(rows, allow_partial=False):
    """Scene medians -> parent medians -> work medians -> class medians.

    rows contain dataset, scene_id, parent_id, work_group, instrument, sdr.
    Partial results explicitly have no official overall score.
    """
    parents = defaultdict(list)
    seen = set()
    parent_work = {}
    for row in rows:
        dataset, instrument = row['dataset'], row['instrument']
        if dataset not in EXPECTED_CLASSES or instrument not in EXPECTED_CLASSES[dataset]:
            raise ScoringError(f'Unexpected dataset/instrument: {dataset}/{instrument}')
        unique = (dataset, row['scene_id'], instrument)
        if unique in seen:
            raise ScoringError(f'Duplicate scene-target score: {unique}')
        seen.add(unique)
        parent = str(row['parent_id'])
        key = (dataset, parent)
        if key in parent_work and parent_work[key] != row['work_group']:
            raise ScoringError('One parent cannot belong to multiple work groups')
        parent_work[key] = row['work_group']
        value = float(row['sdr'])
        if not np.isfinite(value) or not -60 <= value <= 60:
            raise ScoringError('Scene scores must be finite and clipped')
        parents[(dataset, instrument, row['work_group'], parent)].append(value)
    works = defaultdict(list)
    for (dataset, instrument, work, parent), values in parents.items():
        works[(dataset, instrument, work)].append(float(np.median(values)))
    instruments = defaultdict(list)
    for (dataset, instrument, work), values in works.items():
        instruments[(dataset, instrument)].append(float(np.median(values)))
    result = {'metric': 'SDR_BSS_Eval_v4_images', 'datasets': {}, 'overall': None}
    complete = True
    for dataset, expected in EXPECTED_CLASSES.items():
        scores = {name: float(np.median(instruments[(dataset, name)]))
                  for name in expected if (dataset, name) in instruments}
        missing = sorted(set(expected) - scores.keys())
        complete &= not missing
        result['datasets'][dataset] = {
            'instruments': scores, 'missing_instruments': missing,
            'score': float(np.mean(list(scores.values()))) if scores else None,
            'complete': not missing}
    if not complete and not allow_partial:
        raise ScoringError('Incomplete class/dataset coverage; use allow_partial for diagnostics')
    if complete:
        result['overall'] = float(np.mean([d['score'] for d in result['datasets'].values()]))
    result['complete'] = complete
    return result


def _path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ScoringError('Audio path escapes root directory')
    return path


def _read_audio(path, reference=False):
    import soundfile as sf
    error = ScoringError if reference else SubmissionError
    try:
        with sf.SoundFile(path) as audio:
            if audio.samplerate != SAMPLE_RATE or audio.channels != 1:
                raise error(f'{path}: expected 44100 Hz mono')
            if audio.format != 'WAV' or audio.subtype != 'FLOAT':
                raise error(f'{path}: expected WAV IEEE float32')
            samples = audio.read(dtype='float64')
    except (OSError, RuntimeError) as exc:
        raise error(f'Cannot read {path}: {exc}') from exc
    if not np.isfinite(samples).all():
        raise error(f'{path}: non-finite audio')
    return samples


def score_files(reference_path, estimate_path, num_samples, min_active_frames=5):
    """Bounded-memory, algebraically equivalent BSS Eval v4 *images* SDR.

    In museval.metrics._bss_decomp_mtifilt:
      e_artif = padded_estimate - s_true - e_spat - e_interf.
    In _bss_crit(..., bsseval_sources_version=False):
      SDR = 10 log10(||s_true||² / ||e_spat+e_interf+e_artif||²).
    Hence the denominator is ||estimate-reference||². The common 511-sample
    convolution tail is zero in this sum. Global filters cancel, so streaming
    does NOT refit filters or switch to BSS Eval v3. This shortcut applies only
    to images SDR, NOT sources SDR, SI-SDR, ISR, SIR or SAR.
    Reference: sigsep/sigsep-mus-eval, museval/metrics.py, version 0.4.1.
    """
    from contextlib import ExitStack

    import soundfile as sf
    with ExitStack() as stack:
        files = []
        for path, is_reference in ((reference_path, True), (estimate_path, False)):
            error = ScoringError if is_reference else SubmissionError
            try:
                audio = stack.enter_context(sf.SoundFile(path))
            except (OSError, RuntimeError) as exc:
                raise error(f'Cannot read {path}: {exc}') from exc
            if audio.samplerate != SAMPLE_RATE or audio.channels != 1:
                raise error(f'{path}: expected 44100 Hz mono')
            if audio.format != 'WAV' or audio.subtype != 'FLOAT':
                raise error(f'{path}: expected WAV IEEE float32')
            if audio.frames != num_samples or num_samples <= 0:
                raise error(f'{path}: length differs from manifest')
            files.append(audio)
        values = []
        for start in range(0, num_samples, SAMPLE_RATE):
            reference = files[0].read(SAMPLE_RATE, dtype='float64')
            estimate = files[1].read(SAMPLE_RATE, dtype='float64')
            expected = min(SAMPLE_RATE, num_samples-start)
            if len(reference) != expected or not np.isfinite(reference).all():
                raise ScoringError('Reference has nonfinite or truncated audio')
            if len(estimate) != expected or not np.isfinite(estimate).all():
                raise SubmissionError('Estimate has nonfinite or truncated audio')
            energy = float(np.sum(reference ** 2))
            # Divide by a full window to include the implicit zero-padded tail.
            if energy / SAMPLE_RATE <= 1e-12:
                values.append(None)
                continue
            if not np.any(estimate):
                values.append(-60.0)
                continue
            error_energy = float(np.sum((estimate-reference) ** 2))
            value = 60.0 if error_energy == 0 else float(10*np.log10(energy/error_energy))
            if np.isnan(value):
                raise ScoringError('NaN SDR on active reference')
            values.append(float(np.clip(value, -60, 60)))
    active = [v for v in values if v is not None]
    if len(active) < min_active_frames:
        raise ScoringError(f'Reference has fewer than {min_active_frames} active frames')
    return {'sdr': float(np.median(active)), 'active_frames': len(active),
            'total_frames': len(values), 'frame_sdr': values}


def score_dataset(manifest, references_root, estimates_root, split=None,
                  allow_partial=False, min_active_frames=5, show_progress=False):
    """Score a prepared {scenes:[...]} manifest; paths are relative to root.

    Required scene keys: scene_id, dataset, parent_id, work_group,
    target_classes, references ({class: relative WAV path}), num_samples,
    sample_rate. Optional split can select a named partition.
    Predictions: <estimates_root>/<scene_id>/<class>.wav.
    """
    if not isinstance(manifest, dict):
        manifest = json.loads(Path(manifest).read_text())
    selected = [scene for scene in manifest['scenes']
                if split is None or scene.get('split') == split]
    actual_ids = {scene['scene_id'] for scene in selected}
    expected_ids = manifest.get('expected_scene_ids')
    missing_ids, unexpected_ids = [], []
    if expected_ids is not None:
        if not expected_ids or len(expected_ids) != len(set(expected_ids)):
            raise ScoringError('Expected scene IDs must be nonempty and unique')
        missing_ids = sorted(set(expected_ids) - actual_ids)
        unexpected_ids = sorted(actual_ids - set(expected_ids))
    scenes_complete = (manifest.get('complete', True) is True
                       and not missing_ids and not unexpected_ids)
    if not scenes_complete and not allow_partial:
        raise ScoringError('Incomplete evaluation manifest or scene coverage; '
                           'use allow_partial for diagnostics')
    rows = []
    seen = set()
    with tqdm(total=sum(len(s['target_classes']) for s in selected), desc='SDR scoring',
              unit='target', disable=not show_progress, mininterval=1.0, dynamic_ncols=True) as progress:
        for scene in selected:
            scene_id = scene['scene_id']
            if scene_id in seen:
                raise ScoringError(f'Duplicate scene ID: {scene_id}')
            seen.add(scene_id)
            targets = scene['target_classes']
            if not targets or len(set(targets)) != len(targets):
                raise ScoringError('Empty or duplicate target classes')
            if scene['sample_rate'] != SAMPLE_RATE:
                raise ScoringError('Manifest sample rate must be 44100')
            for target in targets:
                row = {k: scene[k] for k in ('scene_id', 'dataset', 'parent_id', 'work_group')}
                row.update(instrument=target, **score_files(
                    _path(references_root, scene['references'][target]),
                    _path(estimates_root, f'{scene_id}/{target}.wav'),
                    scene['num_samples'], min_active_frames=min_active_frames))
                rows.append(row)
                progress.update(1)
    if not rows:
        raise ScoringError('No matching evaluation scenes')
    summary = aggregate_scores(rows, allow_partial=allow_partial)
    summary['scene_coverage'] = {
        'complete': scenes_complete, 'evaluated_count': len(seen),
        'expected_count': len(expected_ids) if expected_ids is not None else None,
        'missing_scene_ids': missing_ids, 'unexpected_scene_ids': unexpected_ids,
    }
    if not scenes_complete:
        summary['complete'] = False
        summary['overall'] = None
    summary['scene_scores'] = rows
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--references-root', required=True)
    parser.add_argument('--estimates-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    result = score_dataset(args.manifest, args.references_root, args.estimates_root,
                           split=args.split, allow_partial=args.allow_partial)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'scene_scores'}, indent=2))


if __name__ == '__main__':
    main()
