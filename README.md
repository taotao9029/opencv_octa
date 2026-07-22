# Stroke GPU Classification Project

本项目用于从小鼠眼底/OCTA 视频中提取血管微循环事件特征，并训练/推理脑卒中二分类模型。当前整理后的主线流程是：视频取清晰帧 -> opencv血管分割 -> 视频事件流生成 -> 血管区域事件过滤 -> opencv核心血流特征提取 -> 工程特征 -> RandomForest 分类 。

## 当前目录结构

```
├── class.py                # 随机森林分类训练、评估、模型保存主程序
├── filter_event.py         # 血流事件筛选预处理脚本
├── get_feature.py          # 时序特征提取脚本
├── seg.py                  # OCTA血管分割相关脚本
├── README.md               # 项目说明文档
├── feature_output/         # 特征中间输出目录
├── output_filter_event/    # 筛选后事件输出
├── output_seg/             # 分割结果输出
├── output_stream/          # 原始血流时序流输出
└── models/                 # 模型、特征文件、转换器保存目录
    ├── features_summary.csv                # 原始汇总特征文件
    ├── augmented_full_features.csv         # 数据增强+RFE筛选后全量数据集
    ├── standard_scaler.pkl                 # StandardScaler标准化转换器
    ├── power_transformer.pkl               # Yeo-Johnson幂变换
    ├── rfe_selector.pkl                    # RFE特征筛选器
    ├── rf_7raw_target0.90.pkl              # 训练完成随机森林模型
    ├── baseline_rf_7raw_0.90_metric_summary.csv       # 指标汇总
    └── baseline_rf_7raw_0.90_testset_pred_detail.csv # 测试集样本预测明细
```

