# 평가 계약

이 문서는 현재 `scoring.py`, `evaluation_metrics.py`, `runtime.py`의 동작을
고정해 설명합니다. 파이프라인 도입으로 기존 점수 정의를 변경하지 않습니다.

## 공식 SDR

- `SDR_BSS_Eval_v4_images`를 사용합니다. `museval==0.4.1`, 44,100 Hz mono,
  WAV IEEE float32가 기준이며 예측 길이는 manifest와 같아야 합니다.
- 악기 라벨은 고정합니다. permutation matching을 하지 않습니다.
- 1초 window/hop, 마지막 부분은 zero padding, reference 평균 제곱 에너지
  `> 1e-12`인 frame만 활성 frame입니다. 최소 활성 frame 수는 기본 5입니다.
- 활성 reference에서 완전히 음소거된 예측은 -60 dB, frame 점수 범위는
  [-60, 60] dB입니다. 활성 frame의 NaN은 오류입니다.
- scene-target의 점수는 활성 frame SDR의 중앙값입니다. 이후 같은 parent의
  scene 중앙값 → 같은 work의 parent 중앙값 → 악기별 work 중앙값을 구합니다.
  데이터셋 점수는 악기 점수의 산술평균, overall은 두 데이터셋 점수의 산술평균입니다.
- SynthSOD 15개, URMP 12개 악기 coverage를 확인합니다. manifest의 예상 scene
  목록이 있으면 누락·추가 scene도 확인합니다. 중복 scene/target은 오류입니다.

배열 API는 museval의 전체 곡 고정 필터(길이 512, framewise filters 비활성화)를
참조 구현으로 사용합니다. 파일 API는 images SDR에서 성립하는 대수적 동등성을
사용해 1초씩 읽으며 메모리를 제한합니다. 이 단축식을 SI-SDR나 SIR/SAR로
확장해서는 안 됩니다.

## smoke와 full 구분

`configs/experiments/smoke.yaml`은 짧은 학습·추론의 연결 확인입니다.
`allow_partial: true` 평가 결과는 `diagnostic: true`, 불완전 manifest로 기록되고
공식 `overall`은 없습니다. smoke 점수를 전체 성능으로 비교하지 않습니다.

전체 평가는 `data.smoke: false`, `training.tiny: false`,
`training.steps_per_shard: null`, `evaluation.max_seconds: null`,
`evaluation.allow_partial: false`로 수행합니다. 기본 추론 chunk는 20초이고
Wiener 후처리를 사용합니다. chunk 크기 등 추론 설정은 결과의 provenance에 남습니다.

기본 결과는 `results/validation.json`, 진단 결과는
`results/validation-smoke.json`입니다. 각 결과의 `complete`, `scene_coverage`,
`diagnostic`, checkpoint SHA256, release ID를 함께 확인합니다.
원래 checkpoint와 현재 checkpoint가 다르면 과거 평가 점수를 현재 모델의
성능으로 등록하지 않습니다.

## 예측 재사용과 추가 지표

`evaluation.reuse_predictions: true`는 완료된 full 평가의 checkpoint, release,
scene/target coverage, manifest 및 각 예측 파일 hash와 추론 설정을 검증합니다.
출처 기록이 없는 과거 예측을 새 파이프라인에서 자동으로 신뢰하지 않습니다.
재추론이 필요하면 `reuse_predictions: false`로 명시합니다.

`evaluation.include_metrics: true`는 추가 SIR/SAR 계산을 켭니다. 전체 scene의
모든 reference source를 함께 평가하며 고정 global filter를 사용하므로 RAM과
CPU 비용이 큽니다. scene별 입력 hash와 metric 설정을 키로 캐시합니다.
추가 결과는 `validation-extra.json` 또는 `validation-smoke-extra.json`에 저장합니다.

SIR/SAR은 공식 SDR을 대체하지 않는 진단 지표입니다. reference 또는 estimate의
source 전체가 무음이면 해당 scene의 joint 점수가 정의되지 않을 수 있습니다.
NaN, 양/음의 무한대 개수와 유한 frame coverage를 기록합니다. 일부 유한 frame의
중앙값이나 `valid_instrument_mean`은 부분 진단치이며 완전한 점수로 보고하지 않습니다.
`supplementary`의 `complete`, `instrument_coverage`, `invalid_instruments`를
확인하고 SDR의 scene/class coverage와 함께 판단합니다. 추가 지표의 `complete`만으로
전체 데이터셋 평가 완료를 주장하지 않습니다.

## 비교에 필요한 기록

동일 데이터 버전과 평가 계약으로 실행한 실험만 직접 비교합니다. 코드 snapshot,
최종 설정, 실제 환경, seed, checkpoint hash, 평가 설정과 결과 JSON을 함께
보관합니다. W&B는 이 기록을 조회하는 화면이며 로컬 결과 JSON이 남습니다.
단위 테스트와 smoke는 실행 계약 확인이고, 실제 전체 validation 성능 측정이나
Colab CUDA 검증을 대신하지 않습니다.
