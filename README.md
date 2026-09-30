# YAML + Colab 음원 분리 학습

**새 실험마다 작성할 파일은 `.ipynb`와 `.yaml` 두 개입니다.**
공통 Python 코드는 이 저장소에 한 번 준비되어 있습니다.

1. [사용법 HTML](docs/colab-guide.html)을 브라우저에서 엽니다. 각 코드의 **복사** 버튼을 누르고 Colab 셀에 붙여 넣으세요.
2. 또는 [pipeline.ipynb](pipeline.ipynb)를 Colab에서 열어 1~6번 셀을 실행하세요.
3. 1번 셀의 GitHub 저장소·REF·YAML 경로와 YAML의 GCS 경로·W&B 프로젝트를 지정하세요.

GitHub 코드 → GCP train/validation → YAML·W&B → 학습 → 평가 → W&B 모델 저장.
기존 학습 노트북에는 1~3번을 상단에, 6번을 하단에 붙입니다. 자신의 모델 코드에서
`CHECKPOINT`와 `REPORT_PATH`를 지정하세요. 평가 파일이 없으면 `REPORT_PATH = None`입니다.
사용자 정의 모델 설정은 YAML의 `parameters`에 자유롭게 추가하고 `config["parameters"]`로 읽습니다.
학습 중 지표는 `run.log({"train/loss": float(loss)})`로 기록합니다.

## 처음 한 번 준비

- 이 저장소의 공통 코드·설정·노트북을 GitHub에 올립니다. 이후에는 새 노트북/YAML만 추가합니다.
- GCS에 **가공된** `prepared/train/`과 `prepared/validation/`을 올립니다. 두 폴더에
  `manifest.json`과 해당 음원이 있어야 합니다. DVC 캐시나 raw 데이터는 받지 않습니다.
- Colab Secrets에 `WANDB_API_KEY`를 등록합니다. 비공개 GitHub 저장소에는 `GITHUB_TOKEN`도 필요합니다.
- GCP 데이터 읽기 권한이 있는 계정으로 2번 셀에서 인증합니다.

`smoke.yaml`은 작은 모델로 연결을 확인합니다. `baseline.yaml`과 `log_mel.yaml`은 실제 학습용입니다.
6번 셀로 종료하기 전, 같은 런타임에서 4번 셀 재실행은 로컬 checkpoint에서 이어갑니다.
6번 셀로 종료한 뒤에는 3번 셀에서 새 실험을 시작합니다. 3번 셀 재실행은 새 실험입니다.
Colab이 종료되면 로컬 파일이 사라질 수 있으므로 종료 전에 6번 셀로 모델을 보관하세요.
원격 자동 복구·DVC·CLI 관리 기능은 사용하지 않습니다.

## 코드 위치

- `pipeline.ipynb`: 복사 가능한 준비·학습·평가·저장 셀
- `configs/experiments/`: 실험 YAML
- `competition-kit/ensemble_benchmark/`: 데이터 다운로드, 모델, 학습·평가
- `docs/colab-guide.html`: 복사 버튼이 있는 사용법
- `competition-kit/tests/`: 실행 시 불필요한 개발용 회귀 테스트

## 개발 확인

```bash
python -m pip install -e ".[colab,dev]"
python -m pytest competition-kit/tests -q
ruff check competition-kit/ensemble_benchmark scripts
mypy
python scripts/build_guide.py --check
```

노트북 셀을 수정하면 `python scripts/build_guide.py`로 HTML을 다시 생성합니다.
고정된 전체 의존성 lock 대신 `pyproject.toml` 하나로 설치하며, 실행 환경에 따라 호환 버전이 선택됩니다.
GPU·클라우드 실제 연결은 사용자 계정으로 smoke 실행해 확인합니다.
[평가 정의](docs/evaluation-protocol.md) · [출처와 라이선스](competition-kit/README.md)
