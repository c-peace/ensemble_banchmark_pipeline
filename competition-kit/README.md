# 모델과 데이터 출처

사용법은 [Colab 셀 복사 가이드](../docs/colab-guide.html)를 참고하세요.

## 기준 모델과 조정 사항

공식 SynthSOD X-UMX의 4개 악기군 모델/architecture와 combination magnitude MSE + weighted SDR 목적함수를 사용합니다. Asteroid X-UMX 소스를 커밋 `5fe95b74dfd53186e74ba7d9b85ca4c27f0d9c09`에 고정하고 MIT 고지를 포함했습니다. 현재 PyTorch의 complex STFT/ISTFT에 맞게 수정했습니다.

원 논문 재현과 구분할 조정: 입력 통계 초기값(identity, 학습 가능), 묶음/샘플 학습, gradient clipping5, 학습 설정은 실험 YAML로 지정, 겹치는 waveform 구간 추론. 실제 upstream 평가 코드를 확인한 결과, 4개 모델의 **전체15개 출력을 하나의 joint Wiener1 처리**에 넣은 후 요청된 악기만 반환하도록 구현했습니다. 이전 계획 문서의 악기군별 Wiener 설명은 이 구현으로 정정됩니다.

기본 모델은 약106M parameters입니다. 일반 GPU에서의 실제 실행 시간·최대 메모리는 아직 측정 전입니다. CPU 연결 검사와 full모델 forward 검증은 GPU 학습 성능 보장이 아닙니다. 배포된 사전학습 가중치를 사용하지 않으며, 논문 수치를 재현했다고 주장하지 않습니다. `tiny=True`는 연결 테스트용으로 공식 성능 비교 대상이 아닙니다.

## 공통 평가

- BSS Eval v4 images SDR, 1초 frame/hop, 악기 ID 고정 대응. SDR에서 필터 항이 상쇄되는 동등 수식을 1초 블록으로 계산하여 긴 곡의 메모리를 제한합니다. museval과의 수치 일치 테스트를 포함합니다. SIR/SAR은 선택적 추가 평가로 분리합니다.
- 프레임→부모 녹음→작품 그룹→악기 순서로 중앙값을 집계합니다. 악기 점수의 평균을 자료별 점수로 삼고 SynthSOD15종/URMP12종에 각각50%를 부여합니다.
- 정답 에너지≤1e-12인 프레임 제외, 활성 정답에서 무음 예측−60dB, 값범위±60dB. 유효 프레임5개 미만인 장면은 가공 단계에서 오류로 처리합니다.
- 길이·채널·WAV FLOAT규격·NaN/Inf를 검사합니다. 정답 목록에 장면이 빠졌거나 필수 악기 범주가 없으면 공식 종합 점수를 거부합니다.
- museval의 여러 source 동시 무음 처리 영향을 피하기 위해 target별 SDR을 계산합니다. 실제 joint호출과 SDR 동등성 테스트가 포함되어 있습니다. SIR/SAR은 `evaluation.include_metrics`로 선택하며 [평가 계약](../docs/evaluation-protocol.md)을 따릅니다.
- 노트북 밖에서도 `python -m ensemble_benchmark.scoring --help`로 공통 채점기를 사용할 수 있습니다.

## 원본과 라이선스

- SynthSOD: Garcia-Martinez 등, https://doi.org/10.5281/zenodo.13759492 ; CC BY-SA4.0. 데이터 파생물은 동일 라이선스로 제공하며 mono변환, stem병합, subset선정, 공통gain 및 패키징 변경을 명시합니다. 저자와 원본 링크를 유지하세요. https://creativecommons.org/licenses/by-sa/4.0/
- URMP: Li 등, https://doi.org/10.5061/dryad.ng3r749 ; CC0-1.0. 공식Dryad 직접다운로드가 제한되어 원본출처를 표기한 Kaggle미러 버전1에서26개 검증stem만 확보했습니다. https://www.kaggle.com/datasets/alonhaviv/multi-modal-music-performance-urmp . 미러와 접근불가원본의 전체바이트동일성은 확인하지 않았으며 개별 ZIPCRC·SHA256·48kmono24bit 규격은 검증했습니다.
- SynthSOD-Baseline upstream: https://github.com/repertorium/SynthSOD-Baseline (AGPL3.0). 본 키트의 학습/가공/채점 통합 코드는 AGPL3.0으로 배포하며 vendored Asteroid 파일은 해당 MIT 고지를 유지합니다.
- 파형 청취·최종 품질 승인은 사용자 확인 후 진행합니다. provenance 폴더에 원본 라이선스 메타데이터와 검증 기록이 있습니다.
