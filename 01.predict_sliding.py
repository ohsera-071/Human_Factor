"""
01_predict_trend_sliding.py  —  라벨 안전 슬라이딩 윈도우 + RF/XGBoost 비교
==========================================================================
[이번 추가]
  (B) 라벨 경계를 침범하지 않는 슬라이딩 윈도우
      · 각 1분 구간(label zone) 내부에서만 30초 윈도우를 슬라이딩
      · step=10초 → 구간당 최대 (60-30)/10 + 1 = 4개 윈도우
      · 같은 영상+분(video_minute)을 하나의 그룹으로 묶어 CV에서 섞이지 않게 함
  (C) RF vs XGBoost 비교 + Soft Voting 앙상블
  (D) 피처 중요도 시각화 저장 (feature_importance.png)
  (E) Confusion matrix 시각화 저장 (confusion_matrix.png)
  (F) 피처 추가 전후 CV 비교표 출력

[유지]
  · Leave-One-Video-Out CV (그룹 = video)
  · 프레임 품질 필터(success==1, confidence>=CONF_THRESH)
  · class_weight='balanced'
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
import joblib

warnings.filterwarnings('ignore')

# ================================================================
# ★ 설정
# ================================================================

VIDEO_LABELS = {
    'testvideo_1.csv':  {0:2, 1:1, 2:2, 3:2, 4:2},
    'testvideo_2.csv':  {0:0, 1:1, 2:1, 3:2, 4:2},
    'testvideo_5.csv':  {0:0, 1:1, 2:1, 3:1, 4:0},
    'testvideo_6.csv':  {0:1, 1:1, 2:2, 3:1, 4:0},
    'testvideo_7.csv':  {0:2, 1:2, 2:1, 3:2, 4:0},
    'testvideo_14.csv': {0:1, 1:0, 2:0, 3:1, 4:0},
}

PREDICT_FILE   = 'testvideo_9.csv'
PREDICT_LABELS = {0:0, 1:0, 2:1, 3:0, 4:0}

CONF_THRESH  = 0.90
MIN_FRAMES   = 50       # 슬라이딩 윈도우는 30초짜리 → 기준 낮춤 (원래 150은 60초 기준)

WINDOW_SEC   = 30       # 슬라이딩 윈도우 크기
STEP_SEC     = 10       # 슬라이딩 간격

SAVE_MODEL   = True
MODEL_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_trend.pkl')
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')

LABEL_MAP = {0: 'Low', 1: 'Medium', 2: 'High'}

# ── 피처 세트 정의 (비교용) ──
BASE_FEATURES  = [
    'blink_freq', 'blink_intensity',
    'gaze_x_mean', 'gaze_y_mean',
    'gaze_x_std',  'gaze_y_std',
    'gaze_x_range','gaze_y_range',
]
TREND_FEATURES = ['blink_trend', 'gaze_x_trend', 'gaze_y_trend']
ALL_FEATURES   = BASE_FEATURES + TREND_FEATURES


# ================================================================
# 내부 함수
# ================================================================

def _trend(series):
    y = series.to_numpy(dtype=float)
    if len(y) < 2:
        return 0.0
    x = np.linspace(0.0, 1.0, len(y))
    return float(np.polyfit(x, y, 1)[0])


def _resolve_data_path(filename):
    """파일명을 DATA_DIR 기준 절대경로로 변환."""
    if os.path.isabs(filename):
        return filename
    return os.path.join(DATA_DIR, filename)


def _load_clean(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    if {'success', 'confidence'}.issubset(df.columns):
        df = df[(df['success'] == 1) & (df['confidence'] >= CONF_THRESH)].copy()
    df['video'] = os.path.basename(filepath)
    return df


def _extract_window(g):
    """단일 윈도우 DataFrame → 피처 dict"""
    g = g.sort_values('timestamp')
    return {
        'blink_freq':      g['AU45_c'].mean(),
        'blink_intensity': g['AU45_r'].mean(),
        'gaze_x_mean':     g['gaze_angle_x'].mean(),
        'gaze_y_mean':     g['gaze_angle_y'].mean(),
        'gaze_x_std':      g['gaze_angle_x'].std(),
        'gaze_y_std':      g['gaze_angle_y'].std(),
        'gaze_x_range':    g['gaze_angle_x'].max() - g['gaze_angle_x'].min(),
        'gaze_y_range':    g['gaze_angle_y'].max() - g['gaze_angle_y'].min(),
        'blink_trend':     _trend(g['AU45_c']),
        'gaze_x_trend':    _trend(g['gaze_angle_x']),
        'gaze_y_trend':    _trend(g['gaze_angle_y']),
    }


def sliding_windows(df, labels, with_label=True):
    """
    각 1분 라벨 구간 내에서만 WINDOW_SEC 크기로 슬라이딩.
    경계를 절대 침범하지 않으므로 라벨 오염 없음.
    그룹 키 = video_minute (CV에서 같은 구간끼리 묶임).
    """
    rows = []
    video = df['video'].iloc[0]

    for minute, label in labels.items():
        zone_start = minute * 60
        zone_end   = (minute + 1) * 60

        # 이 구간의 프레임
        zone_df = df[(df['timestamp'] >= zone_start) & (df['timestamp'] < zone_end)]

        # 윈도우 시작점: zone 내부에서 WINDOW_SEC 크기가 들어갈 수 있는 범위
        win_starts = np.arange(zone_start, zone_end - WINDOW_SEC + 1e-9, STEP_SEC)

        for ws in win_starts:
            we = ws + WINDOW_SEC
            win_df = zone_df[(zone_df['timestamp'] >= ws) & (zone_df['timestamp'] < we)]
            if len(win_df) < MIN_FRAMES:
                continue
            row = _extract_window(win_df)
            row['video']        = video
            row['minute']       = minute
            row['video_minute'] = f"{video}_min{minute}"   # 세분화 그룹
            row['win_start']    = round(ws, 1)
            if with_label:
                row['engagement'] = label
            rows.append(row)

    return pd.DataFrame(rows)


def load_and_extract(filepath, labels):
    df = _load_clean(filepath)
    return sliding_windows(df, labels, with_label=True)


def load_predict_only(filepath, labels=None):
    df = _load_clean(filepath)
    if labels:
        return sliding_windows(df, labels, with_label=True)
    # 라벨 없을 때: 1분 구간을 자동 탐지
    df['minute'] = (df['timestamp'] // 60).astype(int)
    fake_labels = {m: -1 for m in df['minute'].unique()}
    return sliding_windows(df, fake_labels, with_label=False)


# ================================================================
# 1. 데이터 로드
# ================================================================

print("=" * 60)
print("[ STEP 1 ] 슬라이딩 윈도우 피처 추출")
print(f"  윈도우: {WINDOW_SEC}s  /  스텝: {STEP_SEC}s  /  라벨 경계 준수: ✅")
print("=" * 60)

all_feats = []
for fn, labels in VIDEO_LABELS.items():
    full_path = _resolve_data_path(fn)
    if os.path.exists(full_path):
        feat = load_and_extract(full_path, labels)
        all_feats.append(feat)
        print(f"  {fn}: {len(feat)}윈도우 (1분 구간 {len(labels)}개 × 최대 4)")
    else:
        print(f"  ❌ {fn} 없음 ({full_path})")

if not all_feats:
    sys.exit("학습할 영상이 없습니다.")

feat_df = pd.concat(all_feats, ignore_index=True)
feat_df = feat_df.dropna(subset=ALL_FEATURES).reset_index(drop=True)
print(f"\n  총 윈도우: {len(feat_df)}개 (원래 1분 단위: ~{len(feat_df)//4}개 수준)")

X_all    = feat_df[ALL_FEATURES]
y_all    = feat_df['engagement']
# CV 그룹: 영상 단위 (같은 영상의 윈도우가 train/test 동시에 안 들어가게)
groups_video = feat_df['video']


# ================================================================
# 2. 피처 세트 비교 (Base vs Base+Trend)
# ================================================================

print("\n" + "=" * 60)
print("[ STEP 2 ] 피처 추가 전후 CV 비교")
print("=" * 60)

logo = LeaveOneGroupOut()

rf_base  = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf_trend = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')

sc_base  = cross_val_score(rf_base,  feat_df[BASE_FEATURES], y_all,
                            groups=groups_video, cv=logo)
sc_trend = cross_val_score(rf_trend, feat_df[ALL_FEATURES],  y_all,
                            groups=groups_video, cv=logo)

print(f"\n  {'피처 세트':<22} {'평균 정확도':>10}  영상별")
print(f"  {'-'*50}")
print(f"  {'Base (8개)':<22} {sc_base.mean():.1%}      "
      + "  ".join(f"{s:.0%}" for s in sc_base))
print(f"  {'Base + Trend (11개)':<22} {sc_trend.mean():.1%}      "
      + "  ".join(f"{s:.0%}" for s in sc_trend))


# ================================================================
# 3. 모델 비교 (RF / XGBoost / Voting)
# ================================================================

print("\n" + "=" * 60)
print("[ STEP 3 ] RF vs XGBoost 비교")
print("=" * 60)

try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=42, verbosity=0
    )
    sc_xgb = cross_val_score(xgb, X_all, y_all, groups=groups_video, cv=logo)
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  ⚠ xgboost 미설치 → pip install xgboost  (RF만 사용)")

rf_final = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
sc_rf    = cross_val_score(rf_final, X_all, y_all, groups=groups_video, cv=logo)

print(f"\n  {'모델':<25} {'평균 CV 정확도':>14}")
print(f"  {'-'*40}")
print(f"  {'Random Forest':<25} {sc_rf.mean():.1%}")

best_model = rf_final
best_name  = 'RandomForest'

if HAS_XGB:
    print(f"  {'XGBoost':<25} {sc_xgb.mean():.1%}")
    # Soft Voting
    voting = VotingClassifier(
        estimators=[('rf', rf_final), ('xgb', xgb)],
        voting='soft'
    )
    sc_vote = cross_val_score(voting, X_all, y_all, groups=groups_video, cv=logo)
    print(f"  {'Soft Voting (RF+XGB)':<25} {sc_vote.mean():.1%}")

    scores = {'RandomForest': sc_rf.mean(), 'XGBoost': sc_xgb.mean(), 'Voting': sc_vote.mean()}
    best_name  = max(scores, key=scores.get)
    best_model = {'RandomForest': rf_final, 'XGBoost': xgb, 'Voting': voting}[best_name]
    print(f"\n  ✅ 최적 모델: {best_name} ({scores[best_name]:.1%})")


# ================================================================
# 4. 최종 모델 학습 + Confusion Matrix
# ================================================================

print("\n" + "=" * 60)
print(f"[ STEP 4 ] 최종 모델 학습: {best_name}")
print("=" * 60)

best_model.fit(X_all, y_all)

y_oof = cross_val_predict(
    RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
    X_all, y_all, groups=groups_video, cv=logo
)
cls = sorted(pd.unique(y_all))
print("\n  [혼동행렬]  행=실제 / 열=예측")
cm = confusion_matrix(y_all, y_oof, labels=cls)
print("         " + "  ".join(f"{LABEL_MAP[c][:3]:>5}" for c in cls))
for c, rowv in zip(cls, cm):
    print(f"  {LABEL_MAP[c][:3]:>4}  " + "  ".join(f"{x:>5}" for x in rowv))
print("\n  [분류 리포트]")
print(classification_report(
    y_all, y_oof, labels=cls,
    target_names=[LABEL_MAP[c] for c in cls],
    zero_division=0
))

if SAVE_MODEL:
    joblib.dump(best_model, MODEL_PATH)
    print(f"  모델 저장: {MODEL_PATH}")


# ================================================================
# 5. 시각화 — 피처 중요도 + Confusion Matrix
# ================================================================

print("\n" + "=" * 60)
print("[ STEP 5 ] 시각화 저장")
print("=" * 60)

BG, CARD = '#0C0C0F', '#14141A'
BORDER, TEXT, MUTED = '#2A2A35', '#E8E8F0', '#666680'
CLR = ['#2DD4A0', '#F5A623', '#F06060']   # High / Med / Low 순

fig = plt.figure(figsize=(14, 10), facecolor=BG)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── (A) 피처 중요도 (RF만) ──
ax_fi = fig.add_subplot(gs[0, :])
ax_fi.set_facecolor(CARD)

if hasattr(best_model, 'feature_importances_'):
    fi_model = best_model
elif hasattr(best_model, 'estimators_'):          # VotingClassifier
    fi_model = best_model.estimators_[0]
else:
    fi_model = None

if fi_model is not None and hasattr(fi_model, 'feature_importances_'):
    fi = fi_model.feature_importances_
    feat_names = ALL_FEATURES
    sorted_idx = np.argsort(fi)[::-1]
    colors_fi  = ['#2DD4A0' if f in TREND_FEATURES else '#5B8DEF'
                  for f in [feat_names[i] for i in sorted_idx]]
    bars = ax_fi.bar(range(len(fi)), fi[sorted_idx], color=colors_fi, width=0.65)
    ax_fi.set_xticks(range(len(fi)))
    ax_fi.set_xticklabels([feat_names[i] for i in sorted_idx],
                           rotation=30, ha='right', color=MUTED, fontsize=9)
    ax_fi.set_title('Feature Importance (RF)', color=TEXT, fontsize=12, pad=10)
    ax_fi.set_ylabel('Importance', color=MUTED, fontsize=9)
    ax_fi.tick_params(colors=MUTED)
    ax_fi.grid(axis='y', color=BORDER, linewidth=0.5)
    for spine in ax_fi.spines.values():
        spine.set_edgecolor(BORDER)
    # 범례
    from matplotlib.patches import Patch
    legend_els = [Patch(color='#2DD4A0', label='Trend feature (추가)'),
                  Patch(color='#5B8DEF', label='Base feature')]
    ax_fi.legend(handles=legend_els, facecolor=CARD, edgecolor=BORDER,
                  labelcolor=TEXT, fontsize=9)

# ── (B) Confusion Matrix ──
ax_cm = fig.add_subplot(gs[1, 0])
ax_cm.set_facecolor(CARD)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
im = ax_cm.imshow(cm_norm, cmap='YlOrRd', vmin=0, vmax=1)
ax_cm.set_xticks(range(len(cls)))
ax_cm.set_yticks(range(len(cls)))
lbl = [LABEL_MAP[c] for c in cls]
ax_cm.set_xticklabels(lbl, color=MUTED)
ax_cm.set_yticklabels(lbl, color=MUTED)
ax_cm.set_xlabel('Predicted', color=MUTED, fontsize=9)
ax_cm.set_ylabel('Actual', color=MUTED, fontsize=9)
ax_cm.set_title('Confusion Matrix (LOVO CV)', color=TEXT, fontsize=11, pad=8)
for i in range(len(cls)):
    for j in range(len(cls)):
        ax_cm.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.0%})",
                   ha='center', va='center',
                   color='#0C0C0F' if cm_norm[i,j] > 0.5 else TEXT,
                   fontsize=9, fontweight='bold')
plt.colorbar(im, ax=ax_cm, fraction=0.046)

# ── (C) CV 정확도 비교 바 ──
ax_bar = fig.add_subplot(gs[1, 1])
ax_bar.set_facecolor(CARD)
compare_labels = ['Base\n(8 feats)', 'Base+Trend\n(11 feats)']
compare_scores = [sc_base.mean(), sc_trend.mean()]
compare_colors = ['#5B8DEF', '#2DD4A0']
if HAS_XGB:
    compare_labels += ['XGBoost', 'Voting']
    compare_scores += [sc_xgb.mean(), sc_vote.mean()]
    compare_colors += ['#F5A623', '#F06060']

brs = ax_bar.bar(compare_labels, compare_scores,
                  color=compare_colors, width=0.5)
for b, s in zip(brs, compare_scores):
    ax_bar.text(b.get_x() + b.get_width()/2, s + 0.01,
                f"{s:.1%}", ha='center', color=TEXT, fontsize=10, fontweight='bold')
ax_bar.set_ylim(0, 1.1)
ax_bar.set_title('CV Accuracy Comparison', color=TEXT, fontsize=11, pad=8)
ax_bar.set_ylabel('Accuracy', color=MUTED, fontsize=9)
ax_bar.tick_params(colors=MUTED)
ax_bar.grid(axis='y', color=BORDER, linewidth=0.5)
for spine in ax_bar.spines.values():
    spine.set_edgecolor(BORDER)

fig.suptitle('Engagement Prediction — Model Analysis', color=TEXT, fontsize=14, y=1.01)
out_fig = os.path.join(OUTPUT_DIR, 'model_analysis.png')
fig.savefig(out_fig, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"  시각화 저장: {out_fig}")


# ================================================================
# 6. 새 데이터 예측
# ================================================================

predict_path = _resolve_data_path(PREDICT_FILE)
print(f"\n{'=' * 60}")
print(f"[ STEP 6 ] 예측: {PREDICT_FILE}")
print("=" * 60)

if not os.path.exists(predict_path):
    print(f"  ⚠ {PREDICT_FILE} 없음 — 예측 건너뜀 ({predict_path})")
else:
    feat_new = load_and_extract(predict_path, PREDICT_LABELS) if PREDICT_LABELS \
               else load_predict_only(predict_path)
    feat_new = feat_new.dropna(subset=ALL_FEATURES).reset_index(drop=True)

    X_new  = feat_new[ALL_FEATURES]
    preds  = best_model.predict(X_new)
    probs  = best_model.predict_proba(X_new)
    classes = best_model.classes_
    prob_dict = {c: probs[:, i] for i, c in enumerate(classes)}

    print(f"  {'분':>3}  {'윈도우':>8}  {'예측':>8}  {'확률(L/M/H)':>18}  {'실제':>8}  정오")
    print("  " + "-" * 62)
    correct = 0
    for i, (_, row) in enumerate(feat_new.iterrows()):
        pred_label = LABEL_MAP[preds[i]]
        pL = prob_dict.get(0, np.zeros(len(preds)))[i]
        pM = prob_dict.get(1, np.zeros(len(preds)))[i]
        pH = prob_dict.get(2, np.zeros(len(preds)))[i]
        prob_str = f"{pL:.2f}/{pM:.2f}/{pH:.2f}"
        actual = int(row['engagement']) if 'engagement' in row else -1
        ok = ('✅' if preds[i] == actual else '❌') if actual >= 0 else ''
        if actual >= 0 and preds[i] == actual:
            correct += 1
        print(f"  {int(row['minute']):>3}분  "
              f"{row['win_start']:>5.0f}s~   "
              f"{pred_label:>8}  {prob_str:>18}  "
              f"{LABEL_MAP[actual] if actual >= 0 else '-':>8}  {ok}")

    if PREDICT_LABELS:
        # 분 단위 다수결 집계
        feat_new['pred'] = preds
        majority = feat_new.groupby('minute')['pred'].agg(
            lambda x: x.value_counts().index[0]
        )
        print(f"\n  [분 단위 다수결 집계]")
        correct_min = 0
        for m, p in majority.items():
            actual = PREDICT_LABELS.get(m, -1)
            ok = '✅' if p == actual else '❌'
            if p == actual:
                correct_min += 1
            print(f"    {m}분: {LABEL_MAP[p]:>6}  (실제: {LABEL_MAP[actual]:>6}) {ok}")
        print(f"  분 단위 정확도: {correct_min}/{len(majority)} = {correct_min/len(majority):.1%}")

print("\n완료!")