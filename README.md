# ML_HCI 프로젝트 안내

OpenFace CSV 기반으로 **몰입도(0/1/2)**를 학습/예측/분석하는 스크립트 모음입니다.

## 1) 폴더 구조

- `dataset/`
  - `testvideo_*.csv` 원본 데이터
- `01.predict_trend.py`
  - trend 피처 포함(11개) 모델 학습/예측(RF/XGBoost)
- `01.predict_sliding.py`
  - 30초 슬라이딩 윈도우 + 모델 비교(RF/XGBoost/Voting)
- `02.realtime_skeleton.py`
  - 오프라인 GIF/실시간 시계열 예측

## 2) 라벨 정의

- 라벨 키: `minute` (분 단위, `timestamp // 60`)
- 라벨 값:
  - `0 = Low`
  - `1 = Medium`
  - `2 = High`

## 3) 실행 순서 (권장)

### A. 모델 학습/예측

#### (trend 11피처)

```powershell
python 01.predict_trend.py
```

#### (슬라이딩 윈도우 + 모델 비교)

```powershell
python 01.predict_sliding.py
```

### C. 실시간/오프라인 시계열

```powershell
python 02.realtime_skeleton.py
```

- 기본은 오프라인 데모(`MLHCI_OFFLINE=1`)
- 라이브 모드:

```powershell
$env:MLHCI_OFFLINE="0"
python 02.realtime_skeleton.py
```

## 4) 주요 스크립트 요약

## `01.predict_trend.py`

- 기본 8개 + trend 3개 = 11피처
- 품질 필터:
  - `success == 1`
  - `confidence >= CONF_THRESH`
- `Leave-One-Video-Out` 평가 지원

## `01.predict_sliding.py`

- 1분 라벨 구간 내부에서만 30초 윈도우 슬라이딩(라벨 경계 안전)
- `dataset` 폴더 자동 참조
- Base vs Trend, RF vs XGBoost vs Voting 비교
- `model_analysis.png` 저장

## `02.analysis.py`

- 예측 없이 분석/시각화만 수행
- trend 포함 피처 분석

## `02.realtime_skeleton.py`

- 모델 우선순위:
  - `model_trend.pkl` 우선
  - 없으면 `model_basic.pkl` fallback
- 데이터 경로 자동 탐색:
  - 현재 폴더 → `dataset/` 순서
- 로그에 trend(`blink/gaze_x/gaze_y`) 함께 출력

