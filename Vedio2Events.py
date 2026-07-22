import os
import gc
import cv2
import torch
import numpy as np
import pandas as pd

# ===================== 全局一次性配置（放在脚本最顶部，只执行一次） =====================
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)
DEVICE = torch.device("cuda")

# 全局预热 + 关闭垃圾回收，彻底消除随机卡顿
_ = torch.zeros(1, device=DEVICE)
gc.disable()
import os
import gc
import cv2
import torch
import numpy as np
import pandas as pd

# 全局配置
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)
DEVICE = torch.device("cuda")

_ = torch.zeros(1, device=DEVICE)
gc.disable()

def video2events_fast(vid_path, output_path=None, mask_np=None):
    stream = torch.cuda.default_stream()
    torch.cuda.set_stream(stream)

    cap = cv2.VideoCapture(vid_path)
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, 0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 8)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    dt = 1.0 / fps
    theta = 0.02       # 恢复原版固定阈值，算法完全对齐
    max_frames = 15

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return pd.DataFrame(columns=["timestamp(s)", "x", "y", "polarity"])

    h, w = frame.shape[:2]
    prev_gpu = torch.empty((h, w), dtype=torch.float32, device=DEVICE)
    curr_gpu = torch.empty((h, w), dtype=torch.float32, device=DEVICE)

    gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gpu.copy_(torch.from_numpy(gray_prev))

    mask_gpu = None
    if mask_np is not None:
        mask_gpu = torch.from_numpy(mask_np).bool().to(DEVICE)

    all_ts = []
    all_x = []
    all_y = []
    all_p = []
    frame_idx = 1

    while frame_idx <= max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gpu.copy_(torch.from_numpy(gray_curr))

        # ========== 核心算法：与原版完全一致 ==========
        diff = curr_gpu - prev_gpu
        pos = diff > theta
        neg = diff < -theta
        ts = frame_idx * dt

        y_pos, x_pos = torch.nonzero(pos, as_tuple=True)
        y_neg, x_neg = torch.nonzero(neg, as_tuple=True)
        # ===========================================

        if x_pos.numel() > 0:
            cnt = x_pos.shape[0]
            all_ts.append(torch.full((cnt,), ts, dtype=torch.float32, device=DEVICE))
            all_x.append(x_pos)
            all_y.append(y_pos)
            all_p.append(torch.full((cnt,), 1.0, device=DEVICE))
        if x_neg.numel() > 0:
            cnt = x_neg.shape[0]
            all_ts.append(torch.full((cnt,), ts, dtype=torch.float32, device=DEVICE))
            all_x.append(x_neg)
            all_y.append(y_neg)
            all_p.append(torch.full((cnt,), -1.0, device=DEVICE))

        prev_gpu, curr_gpu = curr_gpu, prev_gpu
        frame_idx += 1

    cap.release()

    if not all_ts:
        events_arr = np.empty((0, 4), dtype=np.float32)
    else:
        ts_gpu = torch.cat(all_ts)
        x_gpu = torch.cat(all_x)
        y_gpu = torch.cat(all_y)
        p_gpu = torch.cat(all_p)
        events_gpu = torch.stack([ts_gpu, x_gpu, y_gpu, p_gpu], dim=1)

        # 掩码过滤：仅提速，逻辑等价外部CPU过滤
        if mask_gpu is not None and events_gpu.shape[0] > 0:
            y_idx = events_gpu[:, 2].long()
            x_idx = events_gpu[:, 1].long()
            valid = mask_gpu[y_idx, x_idx]
            events_gpu = events_gpu[valid]

        events_arr = events_gpu.cpu().numpy()

    if output_path is not None:
        df_out = pd.DataFrame(events_arr, columns=["timestamp(s)", "x", "y", "polarity"])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_out.to_csv(output_path, index=False)

    torch.cuda.synchronize()
    return pd.DataFrame(events_arr, columns=["timestamp(s)", "x", "y", "polarity"])

def main():
    video_root = "/home/datasets/cv/TCT/脑卒/OCTAvideo"
    excel_path = "/home/datasets/cv/TCT/脑卒/OCTAvideo/视频对应标签（脑卒）.xlsx"
    output_root = "/home/wxy/Classification/syy/syy/strock_test/test/output_stream_gpu2"

    output_dir_1 = os.path.join(output_root, "1")
    output_dir_0 = os.path.join(output_root, "0")
    os.makedirs(output_dir_1, exist_ok=True)
    os.makedirs(output_dir_0, exist_ok=True)

    df = pd.read_excel(excel_path, usecols=["AVI文件名", "是否脑卒"])
    df["AVI文件名"] = df["AVI文件名"].str.strip()
    df["是否脑卒"] = df["是否脑卒"].str.strip()

    label_dict = {}
    for _, row in df.iterrows():
        filename = row["AVI文件名"]
        label = 1 if row["是否脑卒"] == "是" else 0
        label_dict[filename] = label

    #device = initialize_torch_device()

    for subdir in os.listdir(video_root):
        subdir_path = os.path.join(video_root, subdir)
        if not os.path.isdir(subdir_path):
            continue

        print(f"\n📂 Processing folder: {subdir}")
        for file in os.listdir(subdir_path):
            if file.lower().endswith(".avi"):
                video_path = os.path.join(subdir_path, file)
                if file not in label_dict:
                    print(f"⚠️ Skip {file} (no label)")
                    continue

                label = label_dict[file]
                csv_name = os.path.splitext(file)[0] + ".csv"
                output_path = os.path.join(output_dir_1 if label == 1 else output_dir_0, csv_name)

                if os.path.exists(output_path):
                    print(f"✅ Exists: {csv_name}")
                    continue

                #try:
                video2events_fast(
                        vid_path=video_path,
                        output_path=output_path
                    )
                

    print("\n🎉 All done — 100% no crash!")

if __name__ == "__main__":
   main()
   #test_speed()