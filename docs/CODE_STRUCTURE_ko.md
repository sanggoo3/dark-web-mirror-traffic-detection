# 코드 구성과 실행 흐름

## 폴더별 역할

| 경로 | 역할 |
| --- | --- |
| `data/processed/` | 학습에 사용하는 전처리 방향 시퀀스, 상대 시간, 보조 통계 특징 |
| `data/metadata/` | 가명화된 세션·사이트·그룹 배정, 처리시간 측정 표본 인덱스 |
| `data/splits/` | Group-aware / session-stratified의 외부·내부 분할 인덱스 |
| `configs/` | 후보 설정, fold별 선택 설정, 실험 규칙과 특징 스케일러 기록 |
| `src/darkmirror_repro/` | 실제 실행 코드 |
| `results/recorded/` | 기존 실험의 튜닝 기록, validation/test 예측값, 처리시간 원자료 |
| `results/tables/` | 공개 기록을 재집계한 논문 표와 그림 |
| `tests/` | 입력, 분할, 특징, threshold 및 집계 검사 |
| `artifacts/` | 새로 실행한 학습·평가 결과. 실행 시 생성하며 Git 추적 대상에서 제외 |

원시 PCAP은 포함하지 않는다. 기본 학습은 공개된 전처리 시퀀스에서 시작한다.

## Python 모듈별 역할

| 파일 | 하는 일 |
| --- | --- |
| `__main__.py` | 명령어를 받아 검사·튜닝·학습·평가·통계·시간 측정으로 연결 |
| `io.py` | 데이터 및 분할표 읽기, SHA-256 검사, 입력 길이와 handshake 조건 구성 |
| `models.py` | 신경망 9개 계열의 실제 레이어 구조 정의 |
| `engine_core.py` | 공통 신경망 학습, Adam/loss, early stopping, 최적 checkpoint 저장, 지표와 threshold 계산 |
| `engine.py` | CUDA 실행, 모델 생성, RNN 입력 모양 변환, light16 결합, ROCKET-style 특징·분류기, 저장 모델 로딩 |
| `experiments.py` | 내부 validation 튜닝, 최종 seed별 학습, threshold 고정, test 평가 순서 관리 |
| `features.py` | 같은 첫 526개 패킷 구간에서 타이밍·방향 구간 통계 16개 계산 |
| `statistics.py` | seed 평균·SD·신뢰구간, 조건부 domain bootstrap, 대응 비교 계산 |
| `reproduce.py` | 공개된 예측값으로 논문 결과 재계산 또는 새 실험 결과 통계 집계 |
| `preprocessing.py` | 별도 보유 PCAP의 TShark 기준 방향 추출, 절단·패딩, 고속 파서 대조 |
| `direct_parser.py` | 검증된 PCAPNG 형식의 고속 방향·시간 추출. 범용 QUIC 분석기는 아님 |
| `latency.py` | 별도 보유 원시 캡처와 학습된 모델로 단일 세션 end-to-end 시간 측정 |

ROCKET-style은 `models.py`의 신경망이 아니다. `engine.py`에서 고정 랜덤 convolution 특징을 만들고, fit 데이터에만 맞춘 StandardScaler와 LogisticRegression으로 분류한다. canonical ROCKET과 동일한 구현으로 부르면 안 된다.

## 새로 실험할 때

```text
전처리 데이터 + 고정된 분할표 + 후보 설정
    → tune: 내부 validation으로 설정 선택
    → train: 선택 설정으로 seed별 학습, 최적 checkpoint 복원
             최종 validation으로 threshold 선택
             checkpoint·설정·threshold를 test_lock.json에 고정
    → evaluate: 고정된 모델을 test에서 batch=1로 평가
    → summarize: fold별 결과 → seed별 평균 → 평균·SD·신뢰구간
```

튜닝·threshold 선택에 test 결과를 사용하지 않는다. 통계 결합의 스케일러도 fit 데이터에만 맞춘다. 테스트 평가 전에 해당 실행에서 요청한 학습 및 threshold 결정을 먼저 완료한다.

```console
python -m darkmirror_repro tune --out artifacts/rerun
python -m darkmirror_repro train --out artifacts/rerun
python -m darkmirror_repro evaluate --out artifacts/rerun
python -m darkmirror_repro summarize --out artifacts/rerun
```

이미 선택된 설정으로 재학습하려면 `tune`을 생략하고 `train --recorded-choices`를 사용한다. 일부 모델·fold·seed만 실행하려면 README의 필터 옵션을 사용한다.

## 기존 결과만 확인할 때

```console
python -m darkmirror_repro verify
python -m darkmirror_repro reproduce --out artifacts/reproduced
```

`reproduce`는 재학습 명령이 아니다. 공개된 validation/test 확률과 튜닝 기록에서 지표·선택 규칙·통계를 재계산하고, 기존 보고값과 일치하는지 확인한다. 처리시간도 기존 기록을 재집계한다. 새로 측정하려면 학습된 모델과 별도 원시 캡처를 준비하고 `benchmark`를 실행한다.

## 입력 기준

- 주 실험: 20초 세션의 첫 526개 적격 패킷 방향. 초기 20개 포함.
- Post-handshake 대조군: `q[20:526]`, 길이 506. 제거는 한 번만 수행.
- 길이·handshake 조건은 `experiments.py`의 `paper_conditions`와 `io.py`의 `sequence`에서 결정.
- 신경망 입력은 기본적으로 `(N,1,L)`이며, RNN에는 동일한 방향값을 `(N,L,1)`로 바꾸어 전달.
- 기본 direction-only에는 시간·패킷 크기·전체 패킷 수를 추가하지 않는다. 상대 시간은 명시적인 light16 결합 실험에서만 사용.

요약 설명에서 그룹 수 표기를 제거하더라도, 실제 그룹 ID·분할표·누수 검사·통계 계산은 유지한다. 이 문구 수정으로 데이터, 학습 조건, 보고된 성능은 변경되지 않는다.
