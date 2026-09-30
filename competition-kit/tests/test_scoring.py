import numpy as np
import pytest

from ensemble_benchmark.scoring import (
    CLASSES, SAMPLE_RATE, ScoringError, SubmissionError, aggregate_scores,
    score_dataset, score_target, score_files,
)


def signal(seconds=5):
    return np.random.default_rng(19).normal(0, .1, SAMPLE_RATE * seconds)


def test_perfect_and_muted():
    ref = signal()
    assert score_target(ref, ref)['sdr'] == 60
    result = score_target(ref, np.zeros_like(ref))
    assert result['sdr'] == -60
    assert result['active_frames'] == 5


def test_reference_mask_not_prediction_mask_and_tail():
    ref = np.r_[signal(), np.zeros(SAMPLE_RATE), signal()[:1000]]
    est = ref.copy()
    est[:SAMPLE_RATE] = 0
    result = score_target(ref, est)
    assert result['active_frames'] == 6
    assert result['total_frames'] == 7
    assert result['frame_sdr'][0] == -60
    assert result['frame_sdr'][5] is None
    assert result['frame_sdr'][6] is not None


def test_wrong_shape_nan_and_inactive_reference():
    ref = signal()
    for est in (ref[:-1], ref[:, None], ref * np.nan):
        with pytest.raises(SubmissionError):
            score_target(ref, est)
    with pytest.raises(ScoringError):
        score_target(np.zeros_like(ref), ref)


def test_matches_joint_museval_without_permutation():
    from museval.metrics import bss_eval
    rng = np.random.default_rng(22)
    refs = rng.normal(0, .1, (2, SAMPLE_RATE * 5, 1))
    estimates = refs + rng.normal(0, .03, refs.shape)
    joint = bss_eval(refs, estimates, window=SAMPLE_RATE, hop=SAMPLE_RATE,
                     compute_permutation=False, filters_len=512,
                     framewise_filters=False, bsseval_sources_version=False)[0]
    for i in range(2):
        result = score_target(refs[i, :, 0], estimates[i, :, 0])
        np.testing.assert_allclose(result['frame_sdr'], joint[i], atol=1e-9)
    swapped = score_target(refs[0, :, 0], refs[1, :, 0])
    assert swapped['sdr'] < 0


def row(scene, parent, work, score, dataset='SynthSOD', instrument='Violin'):
    return dict(scene_id=scene, parent_id=parent, work_group=work,
                sdr=score, dataset=dataset, instrument=instrument)


def test_balanced_aggregation_not_scene_weighted():
    rows = [row('a', 'p', 'w', 0), row('b', 'p', 'w', 20),
            row('c', 'q', 'w', 30), row('d', 'r', 'z', 0)]
    result = aggregate_scores(rows, allow_partial=True)
    # p=10, q=30 -> w=20; z=0 -> Violin=10.
    assert result['datasets']['SynthSOD']['instruments']['Violin'] == 10
    assert result['overall'] is None
    with pytest.raises(ScoringError):
        aggregate_scores(rows)
    with pytest.raises(ScoringError):
        aggregate_scores(rows + rows[:1], allow_partial=True)


def test_complete_dataset_weighting():
    rows = [row('s', 'p', 'w', 20, instrument=i) for i in CLASSES]
    rows += [row('u', 'p', 'w', 0, dataset='URMP', instrument=i) for i in CLASSES[:12]]
    assert aggregate_scores(rows)['overall'] == 10


def test_file_validation_and_manifest(tmp_path):
    sf = pytest.importorskip('soundfile')
    ref_root, est_root = tmp_path / 'ref', tmp_path / 'est'
    ref_root.mkdir()
    (est_root / 'scene').mkdir(parents=True)
    ref = signal()
    sf.write(ref_root / 'Violin.wav', ref, SAMPLE_RATE, subtype='FLOAT')
    manifest = {'scenes': [dict(scene_id='scene', dataset='SynthSOD',
        parent_id='p', work_group='w', target_classes=['Violin'],
        references={'Violin': 'Violin.wav'}, num_samples=len(ref), sample_rate=SAMPLE_RATE)]}
    with pytest.raises(SubmissionError):
        score_dataset(manifest, ref_root, est_root, allow_partial=True)
    path = est_root / 'scene' / 'Violin.wav'
    sf.write(path, ref, 48000, subtype='FLOAT')
    with pytest.raises(SubmissionError):
        score_dataset(manifest, ref_root, est_root, allow_partial=True)
    sf.write(path, np.zeros_like(ref), SAMPLE_RATE, subtype='FLOAT')
    result = score_dataset(manifest, ref_root, est_root, allow_partial=True)
    assert result['scene_scores'][0]['sdr'] == -60
    assert result['overall'] is None


def test_metric_nan_is_error_infinity_is_clipped(monkeypatch):
    import museval.metrics
    ref = signal()
    def metric(value):
        def run(*args, **kwargs):
            return (np.full((1, 5), value), None, None, None, None)
        return run
    monkeypatch.setattr(museval.metrics, 'bss_eval', metric(np.nan))
    with pytest.raises(ScoringError, match='NaN SDR'):
        score_target(ref, ref)
    monkeypatch.setattr(museval.metrics, 'bss_eval', metric(np.inf))
    assert score_target(ref, ref)['sdr'] == 60
    monkeypatch.setattr(museval.metrics, 'bss_eval', metric(-np.inf))
    assert score_target(ref, ref)['sdr'] == -60


@pytest.mark.parametrize('missing_id,manifest_complete', [(True, True), (False, False)])
def test_full_class_coverage_cannot_hide_incomplete_scenes(monkeypatch, missing_id, manifest_complete):
    import ensemble_benchmark.scoring as scoring
    monkeypatch.setattr(scoring, 'score_files', lambda *args, **kwargs: {
        'sdr': 10., 'active_frames': 5, 'total_frames': 5, 'frame_sdr': [10.] * 5})
    scenes = []
    for dataset, targets in [('SynthSOD', CLASSES), ('URMP', CLASSES[:12])]:
        scenes.append(dict(scene_id=dataset, dataset=dataset, parent_id='p', work_group='w',
            target_classes=list(targets), references={i: f'{i}.wav' for i in targets},
            num_samples=5*SAMPLE_RATE, sample_rate=SAMPLE_RATE, split='validation'))
    expected = ['SynthSOD', 'URMP'] + (['missing'] if missing_id else [])
    manifest = {'scenes': scenes, 'expected_scene_ids': expected, 'complete': manifest_complete}
    with pytest.raises(ScoringError, match='Incomplete evaluation manifest'):
        score_dataset(manifest, '.', '.', split='validation')
    result = score_dataset(manifest, '.', '.', split='validation', allow_partial=True)
    assert all(d['complete'] for d in result['datasets'].values())
    assert result['overall'] is None
    assert result['complete'] is False
    assert result['scene_coverage']['complete'] is False
    assert result['scene_coverage']['missing_scene_ids'] == (['missing'] if missing_id else [])


@pytest.mark.parametrize('variant', ['noise', 'gain', 'delay', 'filter', 'perfect', 'muted', 'partial_silence'])
def test_streaming_files_match_real_museval_images_sdr(tmp_path, variant):
    import soundfile as sf
    from scipy.signal import lfilter
    rng = np.random.default_rng(764)
    ref = rng.normal(0, .1, 5*SAMPLE_RATE + 111).astype(np.float32)
    if variant == 'noise':
        est = ref + rng.normal(0, .02, len(ref)).astype(np.float32)
    elif variant == 'gain':
        est = ref * .35
    elif variant == 'delay':
        est = np.r_[np.zeros(50, dtype=np.float32), ref[:-50]]
    elif variant == 'filter':
        est = lfilter([.7, .2, .1], [1], ref).astype(np.float32)
    elif variant == 'muted':
        est = np.zeros_like(ref)
    elif variant == 'partial_silence':
        ref[SAMPLE_RATE:2*SAMPLE_RATE] = 0
        est = ref.copy()
        est[2*SAMPLE_RATE:3*SAMPLE_RATE] = 0
    else:
        est = ref.copy()
    sf.write(tmp_path/'reference.wav', ref, SAMPLE_RATE, subtype='FLOAT')
    sf.write(tmp_path/'estimate.wav', est, SAMPLE_RATE, subtype='FLOAT')
    streamed = score_files(tmp_path/'reference.wav', tmp_path/'estimate.wav', len(ref))
    oracle = score_target(ref, est)
    assert streamed['active_frames'] == oracle['active_frames']
    assert streamed['total_frames'] == oracle['total_frames']
    for actual, expected in zip(streamed['frame_sdr'], oracle['frame_sdr']):
        if expected is None:
            assert actual is None
        else:
            assert actual == pytest.approx(expected, abs=1e-9)


def test_streaming_checks_nonfinite_even_in_inactive_frame(tmp_path):
    import soundfile as sf
    ref = np.r_[signal(), np.zeros(SAMPLE_RATE)]
    est = ref.copy()
    est[-1] = np.nan
    sf.write(tmp_path/'r.wav', ref, SAMPLE_RATE, subtype='FLOAT')
    sf.write(tmp_path/'e.wav', est, SAMPLE_RATE, subtype='FLOAT')
    with pytest.raises(SubmissionError, match='nonfinite'):
        score_files(tmp_path/'r.wav', tmp_path/'e.wav', len(ref))
