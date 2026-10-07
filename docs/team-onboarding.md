# 팀원 실험 시작 안내

## 1. 담당자에게 계정을 알려주세요

담당자가 GCP 접근 권한을 부여하고 W&B 팀에 초대합니다.

- **GCP:** Colab에서 사용하는 Google 계정 이메일을 알려주세요. 2번 셀 인증에서도 동일한 계정을 선택하세요.
- **W&B:** 자신의 W&B 계정을 알려주고 팀 초대를 수락하세요.

## 2. W&B API 키를 Colab에 등록하세요

1. [W&B 설정](https://wandb.ai/settings)에서 자신의 API 키를 발급받습니다.
2. Colab 왼쪽 **열쇠 아이콘(Secrets)**에서 아래 내용을 등록합니다.

   | 항목 | 설정 |
   |---|---|
   | 이름 | `WANDB_API_KEY` |
   | 값 | 자신의 W&B API 키 |
   | 노트북 액세스 | 켜기 |

API 키는 코드나 YAML에 작성하지 마세요. GCP용 Secret 키는 별도로 필요하지 않습니다.

## 3. YAML에 실험명을 작성하고 실행하세요

사용할 YAML의 `experiment.name`을 변경하고 GitHub에 저장하세요. 나머지 팀 연결 설정은 유지합니다.

```yaml
experiment:
  name: minsu-baseline-v1
```

실험명은 영문·숫자로 시작하고 영문·숫자·하이픈·밑줄만 사용하세요. W&B에는 이름 뒤에 임의 문자열 8자리가 붙습니다.

Colab 1번 셀에서 해당 YAML 경로를 지정한 뒤, **GPU 런타임에서 1~6번 셀을 순서대로 실행**하세요.

```python
CONFIG_FILE = "configs/experiments/baseline.yaml"
```

일반 학습은 `baseline.yaml`, 연결 확인용 테스트는 `smoke.yaml`을 사용합니다.
