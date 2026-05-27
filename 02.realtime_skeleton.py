"""
03_realtime_skeleton.py  —  Real-time engagement time-series skeleton
====================================================================
OpenFace appends rows to CSV → this script polls, predicts, plots.

Usage:
  1. Run 01.predict.py first (creates model.pkl)
  2. Set LIVE_CSV to OpenFace output path
  3. python 03.realtime_skeleton.py

Offline demo (simulated live CSV → animated GIF):
  python 03.realtime_skeleton.py

Live window (OpenFace writing to LIVE_CSV):
  set MLHCI_OFFLINE=0
  python 03.realtime_skeleton.py
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
OFFLINE_DEMO = os.environ.get('MLHCI_OFFLINE', '1') == '1'

import matplotlib
if OFFLINE_DEMO:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# Settings
# ================================================================
LIVE_CSV     = 'live_output.csv'
DEMO_SOURCE  = 'testvideo_9.csv'
MODEL_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_trend.pkl')
WINDOW_SEC   = 30
POLL_SEC     = 5
MAX_DISPLAY  = 20
DEMO_STEP_SEC = 15          # offline: new prediction every N seconds of video
GIF_FPS      = 2            # animation speed (frames per second)
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')

BASE_FEATURE_COLS = [
    'blink_freq', 'blink_intensity',
    'gaze_x_mean', 'gaze_y_mean',
    'gaze_x_std',  'gaze_y_std',
    'gaze_x_range','gaze_y_range',
]
TREND_FEATURE_COLS = ['blink_trend', 'gaze_x_trend', 'gaze_y_trend']
LABEL_MAP = {0: 'Low', 1: 'Medium', 2: 'High'}
COLOR_MAP = {0: '#F06060', 1: '#F5A623', 2: '#2DD4A0'}

BG, CARD, BORDER = '#0C0C0F', '#14141A', '#2A2A35'
TEXT, MUTED = '#E8E8F0', '#666680'

def _trend(series):
    """Window 내 1차 추세(기울기). x=0~1 정규화."""
    y = series.to_numpy(dtype=float)
    if len(y) < 2:
        return 0.0
    x = np.linspace(0.0, 1.0, len(y))
    return float(np.polyfit(x, y, 1)[0])


def _resolve_data_path(path):
    """상대경로면 우선 현재 폴더, 없으면 dataset 폴더를 확인."""
    if os.path.isabs(path):
        return path
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if os.path.exists(local):
        return local
    return os.path.join(DATA_DIR, path)


# ================================================================
# Model
# ================================================================

if not os.path.exists(MODEL_PATH):
    fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_basic.pkl')
    if os.path.exists(fallback):
        MODEL_PATH = fallback
    else:
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run 01.predict_trend.py first "
            f"(or place {fallback})."
        )
model = joblib.load(MODEL_PATH)
MODEL_FEATURE_COLS = list(getattr(model, 'feature_names_in_', BASE_FEATURE_COLS))
print(f"Model loaded: {MODEL_PATH}")
print(f"Model features: {MODEL_FEATURE_COLS}")


def extract_features_from_window(df_window):
    df_window = df_window.copy()
    df_window.columns = df_window.columns.str.strip()

    required = ['AU45_c', 'AU45_r', 'gaze_angle_x', 'gaze_angle_y']
    missing = [c for c in required if c not in df_window.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    feats = {
        'blink_freq':      df_window['AU45_c'].mean(),
        'blink_intensity': df_window['AU45_r'].mean(),
        'gaze_x_mean':     df_window['gaze_angle_x'].mean(),
        'gaze_y_mean':     df_window['gaze_angle_y'].mean(),
        'gaze_x_std':      df_window['gaze_angle_x'].std(),
        'gaze_y_std':      df_window['gaze_angle_y'].std(),
        'gaze_x_range':    df_window['gaze_angle_x'].max() - df_window['gaze_angle_x'].min(),
        'gaze_y_range':    df_window['gaze_angle_y'].max() - df_window['gaze_angle_y'].min(),
        'blink_trend':     _trend(df_window['AU45_c']),
        'gaze_x_trend':    _trend(df_window['gaze_angle_x']),
        'gaze_y_trend':    _trend(df_window['gaze_angle_y']),
    }
    return feats


def predict_from_window(df_window):
    feats = extract_features_from_window(df_window)
    X = pd.DataFrame([feats])[MODEL_FEATURE_COLS]
    pred = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0]
    prob_dict = {LABEL_MAP[c]: round(float(probs[i]), 3)
                 for i, c in enumerate(model.classes_)}
    return pred, prob_dict, feats


def read_live_csv(filepath):
    filepath = _resolve_data_path(filepath)
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        if 'timestamp' not in df.columns or len(df) == 0:
            return None
        return df
    except Exception:
        return None


def get_window(df_full, current_time, window_sec):
    start = current_time - window_sec
    return df_full[df_full['timestamp'] >= start].copy()


def predict_at_time(df_all, end_t):
    """Predict engagement using the last WINDOW_SEC of data up to end_t."""
    partial = df_all[df_all['timestamp'] <= end_t]
    if len(partial) < 10:
        return None
    current_time = float(partial['timestamp'].max())
    df_window = get_window(partial, current_time, WINDOW_SEC)
    if len(df_window) < 10:
        return None
    pred, prob, feats = predict_from_window(df_window)
    return {
        'time': current_time,
        'pred': pred,
        'prob': prob,
        'pL': prob.get('Low', 0),
        'pM': prob.get('Medium', 0),
        'pH': prob.get('High', 0),
        'blink_trend': feats.get('blink_trend', 0.0),
        'gaze_x_trend': feats.get('gaze_x_trend', 0.0),
        'gaze_y_trend': feats.get('gaze_y_trend', 0.0),
    }


# ================================================================
# Plot
# ================================================================

plt.rcParams.update({'font.family': 'DejaVu Sans'})

fig, (ax_main, ax_prob) = plt.subplots(
    2, 1, figsize=(11, 7),
    gridspec_kw={'height_ratios': [3, 1]},
    facecolor=BG
)

time_buf = []
pred_buf = []
pL_buf, pM_buf, pH_buf = [], [], []
bt_buf, gx_t_buf, gy_t_buf = [], [], []
history = []   # snapshots for GIF animation


def draw_chart(t_buf, p_buf, pl, pm, ph, x_max=None):
    """Draw one frame of the time-series chart."""
    ax_main.cla()
    ax_main.set_facecolor(CARD)
    for spine in ax_main.spines.values():
        spine.set_edgecolor(BORDER)

    if t_buf:
        pt_colors = [COLOR_MAP[p] for p in p_buf]
        ax_main.plot(t_buf, p_buf, color=TEXT, linewidth=1.5, zorder=2, alpha=.7)
        ax_main.scatter(t_buf, p_buf, c=pt_colors, s=70, zorder=3, edgecolors='none')
        ax_main.scatter([t_buf[-1]], [p_buf[-1]],
                        c=[COLOR_MAP[p_buf[-1]]],
                        s=130, zorder=4, edgecolors=TEXT, linewidths=1.2)
        ax_main.annotate(
            LABEL_MAP[p_buf[-1]],
            (t_buf[-1], p_buf[-1]),
            xytext=(8, 8), textcoords='offset points',
            color=COLOR_MAP[p_buf[-1]], fontsize=10, fontweight='bold'
        )

    ax_main.set_yticks([0, 1, 2])
    ax_main.set_yticklabels(['Low', 'Medium', 'High'], color=MUTED)
    ax_main.set_xlabel('Time (sec)', color=MUTED, fontsize=9)
    ax_main.set_title('Real-time Engagement (time series)', color=TEXT, fontsize=11, pad=8)
    ax_main.tick_params(colors=MUTED)
    ax_main.set_ylim(-.5, 2.7)
    if x_max is not None:
        ax_main.set_xlim(0, x_max + 5)
    ax_main.grid(color=BORDER, linewidth=.5)

    ax_prob.cla()
    ax_prob.set_facecolor(CARD)
    for spine in ax_prob.spines.values():
        spine.set_edgecolor(BORDER)

    if pl:
        ax_prob.fill_between(t_buf, 0, pl, color='#F06060', alpha=.55, label='Low')
        ax_prob.fill_between(t_buf, pl,
                             [l + m for l, m in zip(pl, pm)],
                             color='#F5A623', alpha=.55, label='Medium')
        ax_prob.fill_between(t_buf,
                             [l + m for l, m in zip(pl, pm)],
                             [l + m + h for l, m, h in zip(pl, pm, ph)],
                             color='#2DD4A0', alpha=.55, label='High')

    ax_prob.set_ylim(0, 1.05)
    if x_max is not None:
        ax_prob.set_xlim(0, x_max + 5)
    ax_prob.set_ylabel('Probability', color=MUTED, fontsize=9)
    ax_prob.tick_params(colors=MUTED)
    ax_prob.grid(color=BORDER, linewidth=.5)
    ax_prob.legend(loc='upper left', facecolor=CARD, edgecolor=BORDER,
                   labelcolor=TEXT, fontsize=8, ncol=3)


def append_prediction(result):
    global time_buf, pred_buf, pL_buf, pM_buf, pH_buf, bt_buf, gx_t_buf, gy_t_buf, history

    time_buf.append(result['time'])
    pred_buf.append(result['pred'])
    pL_buf.append(result['pL'])
    pM_buf.append(result['pM'])
    pH_buf.append(result['pH'])
    bt_buf.append(result.get('blink_trend', 0.0))
    gx_t_buf.append(result.get('gaze_x_trend', 0.0))
    gy_t_buf.append(result.get('gaze_y_trend', 0.0))

    time_buf = time_buf[-MAX_DISPLAY:]
    pred_buf = pred_buf[-MAX_DISPLAY:]
    pL_buf = pL_buf[-MAX_DISPLAY:]
    pM_buf = pM_buf[-MAX_DISPLAY:]
    pH_buf = pH_buf[-MAX_DISPLAY:]
    bt_buf = bt_buf[-MAX_DISPLAY:]
    gx_t_buf = gx_t_buf[-MAX_DISPLAY:]
    gy_t_buf = gy_t_buf[-MAX_DISPLAY:]

    history.append({
        'time': list(time_buf),
        'pred': list(pred_buf),
        'pL': list(pL_buf),
        'pM': list(pM_buf),
        'pH': list(pH_buf),
        'blink_trend': list(bt_buf),
        'gaze_x_trend': list(gx_t_buf),
        'gaze_y_trend': list(gy_t_buf),
    })

    print(f"  [{result['time']:.1f}s]  pred: {LABEL_MAP[result['pred']]}  "
          f"prob: {result['prob']}  "
          f"trend(b/gx/gy): {result.get('blink_trend', 0.0):+.3f}/"
          f"{result.get('gaze_x_trend', 0.0):+.3f}/"
          f"{result.get('gaze_y_trend', 0.0):+.3f}")


def update(_frame):
    """Live mode: read CSV and append one prediction."""
    df_full = read_live_csv(LIVE_CSV)
    if df_full is None or len(df_full) < 10:
        return

    current_time = float(df_full['timestamp'].max())
    df_window = get_window(df_full, current_time, WINDOW_SEC)
    if len(df_window) < 10:
        return

    try:
        pred, prob, feats = predict_from_window(df_window)
    except Exception as e:
        print(f"[predict error] {e}")
        return

    append_prediction({
        'time': current_time,
        'pred': pred,
        'prob': prob,
        'pL': prob.get('Low', 0),
        'pM': prob.get('Medium', 0),
        'pH': prob.get('High', 0),
        'blink_trend': feats.get('blink_trend', 0.0),
        'gaze_x_trend': feats.get('gaze_x_trend', 0.0),
        'gaze_y_trend': feats.get('gaze_y_trend', 0.0),
    })
    draw_chart(time_buf, pred_buf, pL_buf, pM_buf, pH_buf)


def animate_offline_frame(i):
    """GIF frame i: show series growing point by point."""
    snap = history[i]
    draw_chart(snap['time'], snap['pred'], snap['pL'], snap['pM'], snap['pH'],
               x_max=x_max_time)


def save_results():
    if not time_buf:
        return
    out = os.path.join(OUTPUT_DIR, 'realtime_result.csv')
    pd.DataFrame({
        'time_sec': time_buf,
        'pred': pred_buf,
        'pred_label': [LABEL_MAP[p] for p in pred_buf],
        'prob_low': pL_buf,
        'prob_medium': pM_buf,
        'prob_high': pH_buf,
        'blink_trend': bt_buf,
        'gaze_x_trend': gx_t_buf,
        'gaze_y_trend': gy_t_buf,
    }).to_csv(out, index=False)
    print(f"Results saved: {out}")


def run_offline_demo():
    global history, x_max_time

    demo_path = _resolve_data_path(DEMO_SOURCE)
    live_path = _resolve_data_path(LIVE_CSV)
    source = demo_path if os.path.exists(demo_path) else live_path
    if not os.path.exists(source):
        raise FileNotFoundError(
            f"No demo data. Need {DEMO_SOURCE} or {LIVE_CSV}. "
            f"(checked: {demo_path}, {live_path})"
        )

    print("=" * 55)
    print("Offline demo — time-series animation")
    print(f"  Source: {source}")
    print(f"  Window: {WINDOW_SEC}s  /  step: {DEMO_STEP_SEC}s")
    print("=" * 55)

    df_all = pd.read_csv(source)
    df_all.columns = df_all.columns.str.strip()
    x_max_time = float(df_all['timestamp'].max())

    checkpoints = list(np.arange(WINDOW_SEC, x_max_time + 1, DEMO_STEP_SEC))
    if len(checkpoints) > MAX_DISPLAY:
        checkpoints = checkpoints[:MAX_DISPLAY]

    history = []
    for end_t in checkpoints:
        result = predict_at_time(df_all, end_t)
        if result:
            append_prediction(result)

    if not history:
        raise RuntimeError("No predictions generated.")

    # Animated GIF — points appear one by one over time
    gif_path = os.path.join(OUTPUT_DIR, 'realtime_demo.gif')
    ani = animation.FuncAnimation(
        fig, animate_offline_frame,
        frames=len(history),
        interval=1000 // GIF_FPS,
        repeat=True,
        cache_frame_data=False,
    )
    plt.tight_layout()
    ani.save(gif_path, writer=PillowWriter(fps=GIF_FPS))
    print(f"Animation saved: {gif_path}  ({len(history)} frames, {GIF_FPS} fps)")

    # Final static frame
    png_path = os.path.join(OUTPUT_DIR, 'realtime_demo.png')
    animate_offline_frame(len(history) - 1)
    fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"Final frame saved: {png_path}")

    save_results()
    print("\nOpen realtime_demo.gif to watch the series grow over time.")


if __name__ == '__main__':
    print("=" * 55)
    print("Real-time engagement monitor")
    print(f"  Watch file: {LIVE_CSV}")
    print(f"  Window: {WINDOW_SEC}s  /  Poll: every {POLL_SEC}s")
    print("=" * 55)

    if OFFLINE_DEMO:
        run_offline_demo()
    else:
        print("Live mode — close the window to stop.")
        ani = animation.FuncAnimation(
            fig, update,
            interval=POLL_SEC * 1000,
            cache_frame_data=False
        )
        plt.tight_layout()
        plt.show()
        save_results()
