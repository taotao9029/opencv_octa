# ====================== 全量依赖导入 ======================
import os
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.spatial import KDTree
import warnings
warnings.filterwarnings("ignore")

# ====================== 全局路径配置 ======================
BASE_ROOT = "/home/wxy/Classification/syy/syy/strock_test/OCTA_pytorch/cpu"
INPUT_DIR = os.path.join(BASE_ROOT, "output_filter_event", "0")
OUTPUT_DIR = os.path.join(BASE_ROOT, "feature_output", "0")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================== 工具函数定义 ======================
def detect_dominant_period(t_arr, time_bin=0.01):
    """FFT检测血管主导搏动周期、频率、搏动幅度"""
    t_min, t_max = t_arr.min(), t_arr.max()
    duration = t_max - t_min

    if duration < 0.1:
        return 1.0, 1.0, 0.0, 0.1

    bins = np.arange(t_min, t_max + time_bin, time_bin)
    event_counts, _ = np.histogram(t_arr, bins=bins)
    fs = 1 / time_bin
    freqs, power = signal.periodogram(event_counts, fs=fs, scaling='spectrum')

    valid_mask = (freqs >= 0.5) & (freqs <= 2.0)
    valid_freqs = freqs[valid_mask]
    valid_power = power[valid_mask]

    if len(valid_power) == 0:
        return 1.0, 1.0, 0.0, duration

    dom_idx = np.argmax(valid_power)
    dom_freq = valid_freqs[dom_idx]
    dom_period = 1.0 / dom_freq
    pulse_amp = np.sqrt(valid_power[dom_idx])

    return dom_freq, dom_period, pulse_amp, duration

def split_sliding_window(events, time_col, dom_period, overlap_ratio=0.5):
    window_size = 1 * dom_period
    window_step = window_size * (1 - overlap_ratio)
    t_arr = events[time_col].values

    start_times = np.arange(t_arr.min(), t_arr.max() - window_size + 1e-6, window_step)
    window_list = []
    for wid, st in enumerate(start_times):
        et = st + window_size
        win_mask = (events[time_col] >= st) & (events[time_col] < et)
        win_df = events[win_mask].copy()
        win_df['window_id'] = wid
        window_list.append(win_df)

    if not window_list:
        win_df = events.copy()
        win_df['window_id'] = 0
        window_list.append(win_df)

    return pd.concat(window_list, ignore_index=True), len(window_list)

def merge_adjacent_windows(events, merge_step=2):
    wids = sorted(events['window_id'].unique())
    merged_list = []
    new_wid = 0
    for i in range(0, len(wids), merge_step):
        group_wids = wids[i:i+merge_step]
        sub_df = events[events['window_id'].isin(group_wids)].copy()
        sub_df['window_id'] = new_wid
        merged_list.append(sub_df)
        new_wid += 1
    return pd.concat(merged_list, ignore_index=True), new_wid

def calculate_blood_flow_velocity_ransac(window_events, time_col='t',
                                          max_displacement=4, k_neighbors=3, ransac_thresh=2.5):
    """能算出真实速度就返回，算不出返回 (None, None)，不填0"""
    df = window_events.sort_values(time_col).reset_index(drop=True)
    x = df['x'].values.astype(np.float32)
    y = df['y'].values.astype(np.float32)
    t = df[time_col].values

    # 窗口事件太少，直接无效
    if len(df) < 15:
        return None, None

    mid_idx = len(df) // 2
    prev_pts = np.column_stack((x[:mid_idx], y[:mid_idx]))
    curr_pts = np.column_stack((x[mid_idx:], y[mid_idx:]))
    n = min(len(prev_pts), len(curr_pts))
    prev_pts = prev_pts[:n]
    curr_pts = curr_pts[:n]
    dt = t[mid_idx:mid_idx+n] - t[:n]

    kd = KDTree(curr_pts)
    dists, idxs = kd.query(prev_pts, k=k_neighbors)
    dists = dists[:, 0]
    idxs = idxs[:, 0]

    valid_mask = dists < max_displacement
    if np.sum(valid_mask) < 4:
        return None, None

    m_prev = prev_pts[valid_mask]
    m_curr = curr_pts[idxs[valid_mask]]
    dt_sel = dt[valid_mask]

    def ransac_filter(prev, curr, dt_arr, th):
        max_iter = 50
        best_in = np.zeros(len(prev), dtype=bool)
        for _ in range(max_iter):
            if len(prev) < 4:
                break
            samp_idx = np.random.choice(len(prev), 4, replace=False)
            dx_samp = curr[samp_idx,0] - prev[samp_idx,0]
            dy_samp = curr[samp_idx,1] - prev[samp_idx,1]
            dt_samp = dt_arr[samp_idx]
            vx = np.mean(dx_samp / dt_samp)
            vy = np.mean(dy_samp / dt_samp)
            dx_pred = vx * dt_arr
            dy_pred = vy * dt_arr
            res = np.sqrt((curr[:,0]-prev[:,0]-dx_pred)**2 + (curr[:,1]-prev[:,1]-dy_pred)**2)
            inlier = res < th
            if np.sum(inlier) > np.sum(best_in):
                best_in = inlier
        return best_in

    inlier_mask = ransac_filter(m_prev, m_curr, dt_sel, ransac_thresh)
    m_prev = m_prev[inlier_mask]
    m_curr = m_curr[inlier_mask]
    dt_sel = dt_sel[inlier_mask]

    if len(m_prev) < 3:
        return None, None

    dx = m_curr[:,0] - m_prev[:,0]
    dy = m_curr[:,1] - m_prev[:,1]
    vx = dx / dt_sel
    vy = dy / dt_sel
    spd_arr = np.sqrt(vx**2 + vy**2)

    return np.mean(spd_arr), np.std(spd_arr)

def calculate_window_thd(window_df, time_col='t', time_bin=0.1):
    t = window_df[time_col].values
    if len(t) < 30:
        return np.nan
    bins = np.arange(t.min(), t.max()+time_bin, time_bin)
    cnt, _ = np.histogram(t, bins=bins)
    if len(cnt) < 15:
        return np.nan
    fs = 1 / time_bin
    fft_val = np.fft.fft(cnt)
    fft_freq = np.fft.fftfreq(len(cnt), 1/fs)
    valid_mask = (fft_freq >= 0.5) & (fft_freq <= 2.0) & (fft_freq > 0)
    valid_freq = fft_freq[valid_mask]
    valid_amp = np.abs(fft_val[valid_mask])
    if len(valid_amp) == 0:
        return np.nan
    fund_idx = np.argmax(valid_amp)
    fund_amp = valid_amp[fund_idx]
    if fund_amp < 1e-6:
        return np.nan
    harm_amp = []
    for n in range(2,5):
        h_freq = valid_freq[fund_idx] * n
        match = np.isclose(valid_freq, h_freq, atol=0.15)
        if np.any(match):
            harm_amp.append(np.max(valid_amp[match]))
    thd = np.sqrt(np.sum(np.square(harm_amp))) / fund_amp
    return thd if not np.isnan(thd) and not np.isinf(thd) else np.nan

def process_single_file(file_path):
    filename = os.path.basename(file_path)
    try:
        df = pd.read_csv(file_path)
        # ========== 修复核心：将真实列名 timestamp(s) 重命名为 t ==========
        df.rename(columns={"timestamp(s)": "t"}, inplace=True)
    except Exception as e:
        print(f"读取文件 {filename} 失败: {str(e)}")
        return None

    t_arr = df['t'].values
    total_ev = len(df)

    try:
        dom_freq, dom_period, pulse_amp, duration = detect_dominant_period(t_arr)
    except Exception as e:
        print(f"计算搏动周期失败 {filename}: {str(e)}")
        return None

    event_density = total_ev / duration if duration > 0 else 0

    windowed_events, _ = split_sliding_window(df, time_col='t', dom_period=dom_period)
    merged_events, _ = merge_adjacent_windows(windowed_events, merge_step=2)
    win_groups = merged_events.groupby('window_id')

    # 只存真正算出来的有效速度
    speed_list = []
    thd_list = []

    for _, g in win_groups:
        spd_mean, spd_std = calculate_blood_flow_velocity_ransac(g)
        thd = calculate_window_thd(g)

        # 只有真有效值才加入列表
        if spd_mean is not None and spd_std is not None and not np.isnan(thd):
            speed_list.append(spd_mean)
            thd_list.append(thd)

    # 没有任何有效窗口 → 该样本速度特征留空，不填0
    if len(speed_list) == 0:
        mean_speed = np.nan
        std_speed  = np.nan
    else:
        mean_speed = np.mean(speed_list)
        std_speed  = np.std(speed_list)

    if len(thd_list) == 0:
        mean_thd = np.nan
        std_thd  = np.nan
    else:
        mean_thd = np.mean(thd_list)
        std_thd  = np.mean(thd_list)

    return {
        "filename": filename,
        "mean_speed": round(mean_speed,4) if not np.isnan(mean_speed) else np.nan,
        "std_speed":  round(std_speed,4) if not np.isnan(std_speed) else np.nan,
        "mean_thd":   round(mean_thd,4) if not np.isnan(mean_thd) else np.nan,
        "std_thd":    round(std_thd,4) if not np.isnan(std_thd) else np.nan,
        "pulse_freq": round(dom_freq,4),
        "pulse_amp":  round(pulse_amp,4),
        "event_density": round(event_density,4)
    }

# ====================== 批量主流程 ======================
if __name__ == "__main__":
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
    csv_files.sort()

    all_feat = []
    for f in csv_files:
        res = process_single_file(os.path.join(INPUT_DIR, f))
        if res:
            all_feat.append(res)
            print(f"✅ 处理：{f}")

    if all_feat:
        feat_df = pd.DataFrame(all_feat)
        # 不做任何0填充、不做均值填充，空值就保留NaN
        save_path = os.path.join(OUTPUT_DIR, "seven_features_summary.csv")
        feat_df.to_csv(save_path, index=False, na_rep="")
        print(f"\n🎉 处理完成，共 {len(all_feat)} 个文件")
        print(f"📁 保存：{save_path}")
    else:
        print("❌ 无有效文件")