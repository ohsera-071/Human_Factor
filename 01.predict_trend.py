"""
01_predict.py  —  새 데이터 → 몰입도 예측  (수정판 + 추세 피처 A)
==========================================
[이번 추가]
  (A) 분 안의 '추세' 피처 3개 추가
      · blink_trend / gaze_x_trend / gaze_y_trend
      · 각 분에서 시간에 따른 1차 기울기(≈ 그 분 동안의 총 변화량)
      · +면 증가, -면 감소. 평균만으론 사라지는 '시간 흐름'을 살림.

[유지/이전 수정]
  · 교차검증: 영상 단위(Leave-One-Video-Out) — RUN_CV 토글로 켜고 끔
  · 항상 새로 학습 (옛 모델 자동 재사용 안 함), 모델은 끝에만 저장
  · 프레임 품질 필터(success==1, confidence>=CONF_THRESH) + 프레임 수 가드

[라벨 의미]  키 = 분(minute), 값 = 몰입도 (0=Low / 1=Medium / 2=High)
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    LeaveOneGroupOut, cross_val_score, cross_val_predict
)
from sklearn.metrics import confusion_matrix, classification_report
import joblib

warnings.filterwarnings('ignore')

# ================================================================
# ★ 설정 — 여기만 수정하세요
# ================================================================

VIDEO_LABELS = {
    'testvideo_1.csv': {0:2, 1:1, 2:2, 3:2, 4:2},
    'testvideo_2.csv': {0:0, 1:1, 2:1, 3:2, 4:2},
    'testvideo_5.csv':  {0:0, 1:1, 2:1, 3:1, 4:0},
    'testvideo_6.csv':  {0:1, 1:1, 2:2, 3:1, 4:0},
    'testvideo_14.csv':  {0:1, 1:0, 2:0, 3:1, 4:0},
    'testvideo_9.csv':  {0:0, 1:0, 2:1, 3:0, 4:0},
}

PREDICT_FILE   = 'testvideo_7.csv'
PREDICT_LABELS = {0:2, 1:2, 2:1, 3:2, 4:0}

RUN_CV      = True   # 영상 단위 교차검증 수행 여부 (A 추가 전후 비교하려면 켜둘 것)
CONF_THRESH = 0.90   # 이 신뢰도 미만 프레임은 버림
MIN_FRAMES  = 150    # 분 구간 프레임이 이보다 적으면 통계 신뢰 어려워 제외

SAVE_MODEL = True
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')

# ================================================================
# 내부 함수
# ================================================================

FEATURE_COLS = [
    'blink_freq', 'blink_intensity',
    'gaze_x_mean', 'gaze_y_mean',
    'gaze_x_std',  'gaze_y_std',
    'gaze_x_range', 'gaze_y_range',
    'blink_trend', 'gaze_x_trend', 'gaze_y_trend',   # ← (A) 추세 피처
]
LABEL_MAP = {0: 'Low', 1: 'Medium', 2: 'High'}


def _trend(series):
    """분 안에서 시간에 따른 1차 추세(기울기).
    x를 0~1로 정규화해서 프레임 수가 달라도 비교 가능 → 값 ≈ 그 분 동안의 총 변화량.
    +면 증가(예: 깜빡임이 점점 늘어남), -면 감소."""
    y = series.to_numpy(dtype=float)
    if len(y) < 2:
        return 0.0
    x = np.linspace(0.0, 1.0, len(y))
    return float(np.polyfit(x, y, 1)[0])


def _load_clean(filepath):
    """CSV 로드 → 헤더 공백 제거 → 품질 필터 → minute/video 부여"""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    if {'success', 'confidence'}.issubset(df.columns):
        df = df[(df['success'] == 1) & (df['confidence'] >= CONF_THRESH)].copy()

    df['minute'] = (df['timestamp'] // 60).astype(int)
    df['video'] = os.path.basename(filepath)
    return df


def _extract_features(df, with_label):
    """분 단위 특징 추출 (요약통계 + 추세). with_label=True면 engagement 포함."""
    rows = []
    for (v, m), g in df.groupby(['video', 'minute']):
        if len(g) < MIN_FRAMES:           # 프레임 너무 적은 구간 제외
            continue
        g = g.sort_values('timestamp')    # 추세 계산 위해 시간순 정렬
        row = {
            'video':           v,
            'minute':          int(m),
            # ── 요약 통계 ──
            # 주의: blink_freq 는 'AU45 검출 프레임 비율'(PERCLOS류)이지 분당 횟수가 아님
            'blink_freq':      g['AU45_c'].mean(),
            'blink_intensity': g['AU45_r'].mean(),
            'gaze_x_mean':     g['gaze_angle_x'].mean(),
            'gaze_y_mean':     g['gaze_angle_y'].mean(),
            'gaze_x_std':      g['gaze_angle_x'].std(),
            'gaze_y_std':      g['gaze_angle_y'].std(),
            'gaze_x_range':    g['gaze_angle_x'].max() - g['gaze_angle_x'].min(),
            'gaze_y_range':    g['gaze_angle_y'].max() - g['gaze_angle_y'].min(),
            # ── (A) 추세 (분 동안의 변화 방향/크기) ──
            'blink_trend':     _trend(g['AU45_c']),
            'gaze_x_trend':    _trend(g['gaze_angle_x']),
            'gaze_y_trend':    _trend(g['gaze_angle_y']),
        }
        if with_label:
            row['engagement'] = int(g['engagement'].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def load_and_extract(filepath, labels):
    """라벨 있는 학습/검증용"""
    df = _load_clean(filepath)
    df['engagement'] = df['minute'].map(labels)
    df = df[df['engagement'].notna()].copy()
    return _extract_features(df, with_label=True)


def load_predict_only(filepath):
    """라벨 없는 예측 전용"""
    df = _load_clean(filepath)
    return _extract_features(df, with_label=False)


# ================================================================
# 1. 학습 (항상 새로 학습)
# ================================================================

print("=" * 55)
print("[ STEP 1 ] 모델 학습")
print("=" * 55)

all_feats = []
for fn, labels in VIDEO_LABELS.items():
    if os.path.exists(fn):
        feat = load_and_extract(fn, labels)
        all_feats.append(feat)
        print(f"  로드: {fn} → {len(feat)}구간")
    else:
        print(f"  ❌ {fn} 없음 (건너뜀)")

if not all_feats:
    sys.exit("학습할 영상이 하나도 없습니다. 파일 경로를 확인하세요.")

feat_df_all = pd.concat(all_feats, ignore_index=True)
feat_df_all = feat_df_all.dropna(subset=FEATURE_COLS).reset_index(drop=True)

X = feat_df_all[FEATURE_COLS]
y = feat_df_all['engagement']
groups = feat_df_all['video']
n_videos = groups.nunique()

model = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight='balanced'
)

# ── 영상 단위 교차검증 (켜져 있을 때만) ──
if RUN_CV and n_videos >= 2:
    logo = LeaveOneGroupOut()
    cv_scores = cross_val_score(model, X, y, groups=groups, cv=logo)
    print(f"\n  Leave-One-Video-Out CV 정확도: {cv_scores.mean():.1%}"
          f"  (영상 {n_videos}개 평균 / 구간 {len(feat_df_all)}개)")
    for vid, sc in zip(np.unique(groups), cv_scores):
        print(f"     - {vid:<18} {sc:.1%}")

    y_oof = cross_val_predict(model, X, y, groups=groups, cv=logo)
    cls = sorted(pd.unique(y))
    print("\n  [혼동행렬]  행=실제 / 열=예측")
    cm = confusion_matrix(y, y_oof, labels=cls)
    print("         " + "  ".join(f"{LABEL_MAP[c][:3]:>3}" for c in cls))
    for c, rowv in zip(cls, cm):
        print(f"    {LABEL_MAP[c][:3]:>4}  " + "  ".join(f"{x:>3}" for x in rowv))
    print("\n  [분류 리포트]  (macro avg 의 f1-score 를 핵심 지표로 보세요)")
    print(classification_report(
        y, y_oof, labels=cls,
        target_names=[LABEL_MAP[c] for c in cls],
        zero_division=0
    ))
elif RUN_CV:
    print("\n  ⚠ 영상이 2개 미만이라 교차검증 생략")
else:
    print("\n  (RUN_CV=False → 교차검증 생략)")

model.fit(X, y)
print(f"  학습 완료: 영상 {n_videos}개 / 구간 {len(feat_df_all)}개 / 피처 {len(FEATURE_COLS)}개")

if SAVE_MODEL:
    joblib.dump(model, MODEL_PATH)
    print(f"  모델 저장: {MODEL_PATH}")


# ================================================================
# 2. 새 데이터 예측
# ================================================================

print(f"\n{'=' * 55}")
print(f"[ STEP 2 ] 예측: {PREDICT_FILE}")
print("=" * 55)

if not os.path.exists(PREDICT_FILE):
    sys.exit(f"예측 파일이 없습니다: {PREDICT_FILE}")

if PREDICT_LABELS:
    feat_new = load_and_extract(PREDICT_FILE, PREDICT_LABELS)
    has_label = True
else:
    feat_new = load_predict_only(PREDICT_FILE)
    has_label = False

feat_new = feat_new.dropna(subset=FEATURE_COLS).reset_index(drop=True)
if len(feat_new) == 0:
    sys.exit("예측할 유효한 구간이 없습니다 (품질 필터/프레임 수 확인).")

X_new   = feat_new[FEATURE_COLS]
preds   = model.predict(X_new)
probs   = model.predict_proba(X_new)
classes = model.classes_

prob_dict = {c: probs[:, i] for i, c in enumerate(classes)}
p_low  = prob_dict.get(0, np.zeros(len(preds)))
p_med  = prob_dict.get(1, np.zeros(len(preds)))
p_high = prob_dict.get(2, np.zeros(len(preds)))

header = f"{'분':>4}  {'예측':>8}  {'확률(L/M/H)':>18}"
if has_label:
    header += f"  {'실제':>8}  {'정오'}"
print(header)
print("-" * (60 if has_label else 42))

correct = 0
for i, (_, row) in enumerate(feat_new.iterrows()):
    pred_label = LABEL_MAP[preds[i]]
    prob_str   = f"{p_low[i]:.2f} / {p_med[i]:.2f} / {p_high[i]:.2f}"
    line = f"{int(row['minute']):>4}분  {pred_label:>8}  {prob_str:>18}"
    if has_label:
        actual = int(row['engagement'])
        ok = '✅' if preds[i] == actual else '❌'
        if preds[i] == actual:
            correct += 1
        line += f"  {LABEL_MAP[actual]:>8}  {ok}"
    print(line)

if has_label:
    acc = correct / len(feat_new)
    print(f"\n정확도: {correct}/{len(feat_new)} = {acc:.1%}")
    print("※ 구간 수가 적으면 이 한 영상 정확도보다 STEP 1 의 CV 결과가 더 믿을 만함.")

# ── 결과 저장 ──
result_df = feat_new[['minute'] + FEATURE_COLS].copy()
result_df['pred']       = preds
result_df['pred_label'] = [LABEL_MAP[p] for p in preds]
result_df['prob_low']   = p_low
result_df['prob_med']   = p_med
result_df['prob_high']  = p_high
if has_label:
    result_df['actual']  = feat_new['engagement'].values
    result_df['correct'] = (result_df['pred'] == result_df['actual']).astype(int)

out_csv = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.path.basename(PREDICT_FILE).replace('.csv', '_predictions.csv')
)
result_df.to_csv(out_csv, index=False)
print(f"\n결과 저장: {out_csv}")


# ================================================================
# 3. 예측 함수 (다른 코드에서 import)
# ================================================================

def predict_from_features(
    blink_freq, blink_intensity,
    gaze_x_mean, gaze_y_mean,
    gaze_x_std,  gaze_y_std,
    gaze_x_range, gaze_y_range,
    blink_trend, gaze_x_trend, gaze_y_trend
):
    """특징값 직접 입력 → (pred:int, probs:dict) 반환"""
    X_input = pd.DataFrame([[
        blink_freq, blink_intensity,
        gaze_x_mean, gaze_y_mean,
        gaze_x_std, gaze_y_std,
        gaze_x_range, gaze_y_range,
        blink_trend, gaze_x_trend, gaze_y_trend
    ]], columns=FEATURE_COLS)

    pred = int(model.predict(X_input)[0])
    prob = model.predict_proba(X_input)[0]
    probs = {LABEL_MAP[c]: round(float(prob[i]), 3)
             for i, c in enumerate(model.classes_)}
    return pred, probs


if __name__ == '__main__':
    p, prob = predict_from_features(
        blink_freq=0.08, blink_intensity=0.15,
        gaze_x_mean=-0.06, gaze_y_mean=0.10,
        gaze_x_std=0.07,   gaze_y_std=0.09,
        gaze_x_range=0.25, gaze_y_range=0.30,
        blink_trend=0.02,  gaze_x_trend=-0.01, gaze_y_trend=0.03
    )
    print("\n[함수 직접 호출 예시]")
    print(f"  예측: {LABEL_MAP[p]} ({p})")
    print(f"  확률: {prob}")