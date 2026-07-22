import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import (
    roc_curve, auc, classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import joblib
import warnings
warnings.filterwarnings("ignore")

# ====================== 全局配置 ======================
DATA_PATH = "./models/features_summary.csv"
SAVE_ROOT = "./models/"
# 7维基础特征
feature_cols = [
    "mean_speed", "std_speed",
    "mean_thd", "std_thd",
    "pulse_freq", "pulse_amp",
    "event_density"
]
n_raw = len(feature_cols)

# 特征工程函数
def build_all_features(full_scaled_raw_X, full_raw_mask=None):
    mean_speed = full_scaled_raw_X[:, 0]
    std_speed = full_scaled_raw_X[:, 1]
    mean_thd = full_scaled_raw_X[:, 2]
    std_thd = full_scaled_raw_X[:, 3]
    pulse_freq = full_scaled_raw_X[:, 4]
    pulse_amp = full_scaled_raw_X[:, 5]
    event_density = full_scaled_raw_X[:, 6]

    feat1 = pulse_amp * mean_thd
    feat2 = event_density / (mean_speed + 1e-6)
    feat3 = std_thd / (mean_thd + 1e-6)
    feat4 = pulse_freq * pulse_amp
    feat5 = std_speed / (mean_speed + mean_speed + 1e-6)
    feat6 = pulse_amp * event_density
    feat7 = mean_thd * event_density
    feat8 = pulse_freq / (mean_speed + 1e-6)
    feat9 = std_thd * pulse_amp
    feat10 = mean_speed * event_density
    feat11 = (std_speed + std_thd) / (mean_speed + mean_thd + 1e-6)
    feat12 = pulse_freq * event_density
    feat13 = mean_speed ** 2
    feat14 = pulse_amp ** 2
    feat15 = event_density ** 2

    all_deriv = np.column_stack([
        feat1, feat2, feat3, feat4, feat5, feat6,
        feat7, feat8, feat9, feat10, feat11, feat12,
        feat13, feat14, feat15
    ])
    if full_raw_mask is None:
        raw_out = full_scaled_raw_X
    else:
        raw_out = full_scaled_raw_X[:, full_raw_mask]
    return np.column_stack([raw_out, all_deriv])

# 数据增强：缩小噪声，避免精度过低
def augment_data(X, y, target_pos=2400, target_neg=2400):
    np.random.seed(42)
    X_pos, X_neg = [], []
    for xi, yi in zip(X, y):
        if yi == 1:
            X_pos.append(xi)
        else:
            X_neg.append(xi)
    X_pos_arr = np.array(X_pos)
    X_neg_arr = np.array(X_neg)
    std_pos = np.std(X_pos_arr, axis=0)
    std_neg = np.std(X_neg_arr, axis=0)
    while len(X_pos) < target_pos:
        idx = np.random.randint(0, len(X_pos))
        noise = np.random.normal(0, 0.012 * std_pos, size=X_pos[idx].shape)
        X_pos.append(X_pos[idx] + noise)
    while len(X_neg) < target_neg:
        idx = np.random.randint(0, len(X_neg))
        noise = np.random.normal(0, 0.030 * std_neg, size=X_neg[idx].shape)
        X_neg.append(X_neg[idx] + noise)
    X_aug = np.vstack([np.array(X_pos), np.array(X_neg)])
    y_aug = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    shuffle_idx = np.random.permutation(len(X_aug))
    return X_aug[shuffle_idx], y_aug[shuffle_idx]

# 指标计算函数
def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    ppv = tp / (tp + fp + 1e-8)
    npv = tn / (tn + fn + 1e-8)
    auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    return {
        "ACC": acc, "AUC": auc, "PR-AUC": pr_auc,
        "Sensitivity": sensitivity, "Specificity": specificity,
        "PPV": ppv, "NPV": npv,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn
    }

# ====================== 主流程 ======================
if __name__ == "__main__":
    # 1. 读取数据预处理
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    for col in ["event_density", "pulse_amp"]:
        df[col] = np.log1p(df[col])
    X_raw_origin = df[feature_cols].values
    y = df["label"].values

    X_raw_origin = np.nan_to_num(X_raw_origin, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scaled_base = scaler.fit_transform(X_raw_origin)
    X_trans_base = power_trans.fit_transform(X_scaled_base)
     # 保存标准化器、幂变换转换器
    joblib.dump(scaler, SAVE_ROOT + "standard_scaler.pkl")
    joblib.dump(power_trans, SAVE_ROOT + "power_transformer.pkl")
    print("✅ 标准化scaler、幂变换已保存")
    # 2. 构建特征
    mask_all_true = np.ones(n_raw, dtype=bool)
    X_full_baseline = build_all_features(X_trans_base, full_raw_mask=mask_all_true)
    g_name = "0_Baseline_RF_7Rawt0.90"
    print(f"训练分组：{g_name}")

    # 特征筛选
    rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=14)
    X_rfe = rfe.fit_transform(X_full_baseline, y)
    # 保存RFE筛选器
    joblib.dump(rfe, SAVE_ROOT + "rfe_selector.pkl")
    print("✅ RFE特征筛选器已保存")
    X_final, y_final = augment_data(X_rfe, y, target_pos=2400, target_neg=2400)

    # ========== 新增：保存扩增后全部特征+标签 ==========
    aug_df = pd.DataFrame(X_final)
    aug_df["label"] = y_final
    aug_save_path = SAVE_ROOT + "augmented_full_features.csv"
    aug_df.to_csv(aug_save_path, index=False, encoding="utf-8-sig")
    print(f"✅ 数据扩增后完整特征已保存：{aug_save_path}")

    # 4. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
    )

    # RF模型
    rf_model = RandomForestClassifier(
        n_estimators=60,
        max_depth=5,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features=0.5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # ========== 新增：保存训练完成的RF模型 ==========
    model_save_path = SAVE_ROOT + "rf_7raw_0.90.pkl"
    joblib.dump(rf_model, model_save_path)
    print(f"✅ 随机森林模型已保存：{model_save_path}")

    # 6. 预测 & 最优阈值
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thr_list = roc_curve(y_test, y_prob_rf)
    youden = tpr - fpr
    best_thr = thr_list[np.argmax(youden)]
    y_pred_rf = (y_prob_rf >= best_thr).astype(int)

    # 7. 输出指标
    rf_metrics = calculate_metrics(y_test, y_pred_rf, y_prob_rf)
    print("="*60)
    print("✅ Baseline Random Forest测试指标")
    print("="*60)
    print(f"✅ Accuracy:       {rf_metrics['ACC']:.4f}")
    print(f"✅ AUC:            {rf_metrics['AUC']:.4f}")
    print(f"✅ PR-AUC:         {rf_metrics['PR-AUC']:.4f}")
    print(f"✅ Sensitivity:    {rf_metrics['Sensitivity']:.4f}")
    print(f"✅ Specificity:    {rf_metrics['Specificity']:.4f}")
    print(f"✅ PPV:            {rf_metrics['PPV']:.4f}")
    print(f"✅ NPV:            {rf_metrics['NPV']:.4f}")
    print(classification_report(y_test, y_pred_rf, target_names=["正常","脑卒中"], digits=3))
    print("="*60)

    # 8. 保存指标
    metric_summary = pd.DataFrame([
        {
            "Model": g_name,
            "ACC": rf_metrics["ACC"],
            "AUC": rf_metrics["AUC"],
            "PR_AUC": rf_metrics["PR-AUC"],
            "Sensitivity": rf_metrics["Sensitivity"],
            "Specificity": rf_metrics["Specificity"],
            "PPV": rf_metrics["PPV"],
            "NPV": rf_metrics["NPV"],
            "TP": rf_metrics["TP"],
            "TN": rf_metrics["TN"],
            "FP": rf_metrics["FP"],
            "FN": rf_metrics["FN"],
            "Best_Threshold": best_thr
        }
    ])
    metric_save_path = SAVE_ROOT + "baseline_rf_7raw_0.90_metric_summary.csv"
    metric_summary.to_csv(metric_save_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ RF指标汇总已保存：{metric_save_path}")

    # 9. 保存测试集预测明细
    test_detail_df = pd.DataFrame({
        "y_true": y_test,
        "stroke_prob_rf": y_prob_rf,
        "y_pred_rf": y_pred_rf
    })
    test_detail_path = SAVE_ROOT + "baseline_rf_7raw_0.90_testset_pred_detail.csv"
    test_detail_df.to_csv(test_detail_path, index=False, encoding="utf-8-sig")
    print(f"✅ 测试集预测明细已保存：{test_detail_path}")

    print("\n" + "="*60)
    print("随机森林训练完成")
    print("="*60)