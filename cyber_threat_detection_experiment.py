"""
=============================================================================
AI-Based Cyber Threat Detection System for Enterprise Networks
Full Experiment Pipeline — Journal Paper Results Generator
=============================================================================
Datasets  : CICIDS2017 + UNSW-NB15
Models    : Random Forest | SVM | LSTM | Hybrid CNN-LSTM (Proposed)
Outputs   : Table 3, 4, 5 (CSV) + Figure 1, 2, 3 (PNG) + Full metrics log
Python    : 3.9+
=============================================================================
SETUP INSTRUCTIONS (run once in terminal):
    pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
    pip install imbalanced-learn openpyxl

DATASET DOWNLOAD:
    CICIDS2017 : https://www.unb.ca/cic/datasets/ids-2017.html
                 Download the CSV files (MachineLearningCSV.zip)
                 Place all CSVs in a folder called: data/CICIDS2017/

    UNSW-NB15  : https://research.unsw.edu.au/projects/unsw-nb15-dataset
                 Download UNSW_NB15_training-set.csv + UNSW_NB15_testing-set.csv
                 Place them in a folder called: data/UNSW_NB15/

FOLDER STRUCTURE:
    your_project/
    ├── cyber_threat_detection_experiment.py   ← this file
    ├── data/
    │   ├── CICIDS2017/
    │   │   ├── Monday-WorkingHours.pcap_ISCX.csv
    │   │   ├── Tuesday-WorkingHours.pcap_ISCX.csv
    │   │   └── ... (all CSV files)
    │   └── UNSW_NB15/
    │       ├── UNSW_NB15_training-set.csv
    │       └── UNSW_NB15_testing-set.csv
    └── results/       ← auto-created, all outputs saved here
=============================================================================
"""

# ─── IMPORTS ─────────────────────────────────────────────────────────────────
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble         import RandomForestClassifier
from sklearn.svm              import SVC
from sklearn.preprocessing    import LabelEncoder, MinMaxScaler
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import (accuracy_score, precision_score,
                                       recall_score, f1_score,
                                       roc_auc_score, confusion_matrix,
                                       classification_report, roc_curve)
from imblearn.over_sampling   import SMOTE

import tensorflow as tf
from tensorflow.keras.models  import Sequential, Model
from tensorflow.keras.layers  import (Dense, LSTM, Conv1D, MaxPooling1D,
                                       Flatten, Dropout, Input,
                                       BatchNormalization, Reshape)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils   import to_categorical

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
RESULTS_DIR   = "results"
DATA_CICIDS   = "data/CICIDS2017"
DATA_UNSW     = "data/UNSW_NB15"
TEST_SIZE     = 0.20
VAL_SIZE      = 0.15
MAX_SAMPLES   = 100_000   # Cap per dataset for manageable training time on PC
EPOCHS        = 50
BATCH_SIZE    = 256

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── COLOUR PALETTE ──────────────────────────────────────────────────────────
COLORS = {
    "proposed" : "#1f77b4",
    "rf"       : "#ff7f0e",
    "svm"      : "#2ca02c",
    "lstm"     : "#d62728",
    "accent"   : "#9467bd"
}

# =============================================================================
# SECTION 1 — DATA LOADING
# =============================================================================

def load_cicids2017(data_dir: str) -> pd.DataFrame:
    """Load and merge all CICIDS2017 CSV files."""
    print("\n[1/6] Loading CICIDS2017 dataset...")
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}.\n"
            "Please download CICIDS2017 from:\n"
            "https://www.unb.ca/cic/datasets/ids-2017.html"
        )
    frames = []
    for f in files:
        df = pd.read_csv(os.path.join(data_dir, f), encoding='utf-8', low_memory=False)
        df.columns = df.columns.str.strip()
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    print(f"    CICIDS2017 raw shape: {data.shape}")
    return data


def load_unsw_nb15(data_dir: str) -> pd.DataFrame:
    """Load UNSW-NB15 training and test CSVs."""
    print("[1/6] Loading UNSW-NB15 dataset...")
    train_path = os.path.join(data_dir, "UNSW_NB15_training-set.csv")
    test_path  = os.path.join(data_dir, "UNSW_NB15_testing-set.csv")
    for p in [train_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"File not found: {p}\n"
                "Please download UNSW-NB15 from:\n"
                "https://research.unsw.edu.au/projects/unsw-nb15-dataset"
            )
    data = pd.concat(
        [pd.read_csv(train_path), pd.read_csv(test_path)],
        ignore_index=True
    )
    print(f"    UNSW-NB15 raw shape: {data.shape}")
    return data


# =============================================================================
# SECTION 2 — PREPROCESSING
# =============================================================================

def preprocess_cicids(df: pd.DataFrame):
    """Clean and encode CICIDS2017 for ML."""
    print("[2/6] Preprocessing CICIDS2017...")

    # Standardise label column name
    label_col = ' Label' if ' Label' in df.columns else 'Label'
    df = df.rename(columns={label_col: 'label'})

    # Drop non-numeric and identifier columns
    drop_cols = ['Flow ID', 'Source IP', 'Source Port',
                 'Destination IP', 'Destination Port', 'Timestamp']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Binary label: 0 = BENIGN, 1 = Attack
    df['label'] = df['label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)

    # Numeric coercion and NaN/Inf removal
    df = df.apply(pd.to_numeric, errors='coerce')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Subsample to MAX_SAMPLES (stratified)
    df = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(min(len(x), MAX_SAMPLES // 2), random_state=42)
    ).reset_index(drop=True)

    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values.astype(np.int32)

    return X, y, df.drop('label', axis=1).columns.tolist()


def preprocess_unsw(df: pd.DataFrame):
    """Clean and encode UNSW-NB15 for ML."""
    print("[2/6] Preprocessing UNSW-NB15...")

    # Binary label column
    label_col = 'label' if 'label' in df.columns else 'Label'
    df = df.rename(columns={label_col: 'label'})

    drop_cols = ['id', 'attack_cat']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Encode categorical columns
    for col in df.select_dtypes(include='object').columns:
        if col != 'label':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    df = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(min(len(x), MAX_SAMPLES // 2), random_state=42)
    ).reset_index(drop=True)

    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values.astype(np.int32)

    return X, y, df.drop('label', axis=1).columns.tolist()


def split_and_scale(X, y):
    """Scale features, apply SMOTE, split into train/val/test."""
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train+val vs test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    # Train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=42, stratify=y_tv
    )

    # SMOTE on training set only
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# SECTION 3 — MODEL DEFINITIONS
# =============================================================================

def build_lstm(input_shape):
    """Standalone LSTM model (baseline)."""
    model = Sequential([
        Reshape((input_shape[1], 1), input_shape=input_shape),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ], name="LSTM_Baseline")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_cnn_lstm(input_shape):
    """Proposed Hybrid CNN-LSTM model."""
    inputs = Input(shape=(input_shape[1], 1), name="input_layer")

    # CNN branch — local spatial feature extraction
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # LSTM branch — temporal sequence learning
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(64)(x)
    x = Dropout(0.2)(x)

    # Classification head
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(1, activation='sigmoid', name="output")(x)

    model = Model(inputs, outputs, name="Hybrid_CNN_LSTM_Proposed")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# =============================================================================
# SECTION 4 — TRAINING & EVALUATION
# =============================================================================

def evaluate_sklearn(model, X_test, y_test):
    """Return full metrics dict for sklearn models."""
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    inf_time = (time.perf_counter() - start) * 1000 / len(X_test)  # ms per sample

    y_prob = (model.predict_proba(X_test)[:, 1]
              if hasattr(model, 'predict_proba') else y_pred.astype(float))

    return {
        "Accuracy"    : round(accuracy_score(y_test, y_pred) * 100, 2),
        "Precision"   : round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "Recall"      : round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "F1-Score"    : round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "AUC-ROC"     : round(roc_auc_score(y_test, y_prob) * 100, 2),
        "FPR"         : round((confusion_matrix(y_test, y_pred)[0][1] /
                               max(confusion_matrix(y_test, y_pred)[0].sum(), 1)) * 100, 2),
        "Inf_ms"      : round(inf_time, 4),
        "y_prob"      : y_prob,
        "y_pred"      : y_pred
    }


def evaluate_keras(model, X_test, y_test):
    """Return full metrics dict for Keras models."""
    X_3d = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    start = time.perf_counter()
    y_prob = model.predict(X_3d, verbose=0).flatten()
    inf_time = (time.perf_counter() - start) * 1000 / len(X_test)
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "Accuracy"    : round(accuracy_score(y_test, y_pred) * 100, 2),
        "Precision"   : round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "Recall"      : round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "F1-Score"    : round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "AUC-ROC"     : round(roc_auc_score(y_test, y_prob) * 100, 2),
        "FPR"         : round((confusion_matrix(y_test, y_pred)[0][1] /
                               max(confusion_matrix(y_test, y_pred)[0].sum(), 1)) * 100, 2),
        "Inf_ms"      : round(inf_time, 4),
        "y_prob"      : y_prob,
        "y_pred"      : y_pred
    }


def run_experiment(dataset_name: str, X_train, X_val, X_test,
                   y_train, y_val, y_test):
    """Train all models and collect results for one dataset."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {dataset_name}")
    print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"{'='*60}")

    results   = {}
    histories = {}
    train_times = {}

    # ── Random Forest ────────────────────────────────────────────
    print("\n  [RF] Training Random Forest...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                 max_depth=20, random_state=42)
    rf.fit(X_train, y_train)
    train_times["Random Forest"] = round(time.time() - t0, 1)
    results["Random Forest"] = evaluate_sklearn(rf, X_test, y_test)
    print(f"       F1={results['Random Forest']['F1-Score']}%  "
          f"AUC={results['Random Forest']['AUC-ROC']}%")

    # ── SVM ──────────────────────────────────────────────────────
    print("\n  [SVM] Training SVM (RBF kernel)...")
    t0 = time.time()
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    # SVM is slow on large data — subsample to 30k if needed
    idx = np.random.choice(len(X_train),
                           min(len(X_train), 30_000), replace=False)
    svm.fit(X_train[idx], y_train[idx])
    train_times["SVM"] = round(time.time() - t0, 1)
    results["SVM"] = evaluate_sklearn(svm, X_test, y_test)
    print(f"       F1={results['SVM']['F1-Score']}%  "
          f"AUC={results['SVM']['AUC-ROC']}%")

    callbacks = [
        EarlyStopping(patience=7, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(patience=4, factor=0.5, verbose=0)
    ]
    input_shape = (None, X_train.shape[1])
    X_train_3d  = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val_3d    = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

    # ── LSTM (baseline) ──────────────────────────────────────────
    print("\n  [LSTM] Training LSTM baseline...")
    t0 = time.time()
    lstm = build_lstm(input_shape)
    hist_lstm = lstm.fit(
        X_train_3d, y_train,
        validation_data=(X_val_3d, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=0
    )
    train_times["LSTM"] = round(time.time() - t0, 1)
    histories["LSTM"] = hist_lstm
    results["LSTM"] = evaluate_keras(lstm, X_test, y_test)
    print(f"       F1={results['LSTM']['F1-Score']}%  "
          f"AUC={results['LSTM']['AUC-ROC']}%")

    # ── Proposed: CNN-LSTM ────────────────────────────────────────
    print("\n  [CNN-LSTM] Training proposed Hybrid CNN-LSTM...")
    t0 = time.time()
    cnn_lstm = build_cnn_lstm(input_shape)
    hist_cnnlstm = cnn_lstm.fit(
        X_train_3d, y_train,
        validation_data=(X_val_3d, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=0
    )
    train_times["CNN-LSTM\n(Proposed)"] = round(time.time() - t0, 1)
    histories["CNN-LSTM\n(Proposed)"] = hist_cnnlstm
    results["CNN-LSTM\n(Proposed)"] = evaluate_keras(cnn_lstm, X_test, y_test)
    print(f"       F1={results['CNN-LSTM(Proposed)']['F1-Score']}%  "
          f"AUC={results['CNN-LSTM(Proposed)']['AUC-ROC']}%"
          if "CNN-LSTM(Proposed)" in results else
          f"       F1={list(results.values())[-1]['F1-Score']}%")

    return results, histories, train_times, cnn_lstm, X_test, y_test


# =============================================================================
# SECTION 5 — FIGURE & TABLE GENERATION
# =============================================================================

def save_table3(results_cicids, results_unsw, out_dir):
    """Table 3: Model performance comparison across both datasets."""
    print("\n[5/6] Generating Table 3 — Model Performance Comparison...")
    rows = []
    metric_keys = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", "FPR"]

    for model_name, res in results_cicids.items():
        row = {"Model": model_name, "Dataset": "CICIDS2017"}
        for m in metric_keys:
            row[m] = f"{res[m]}%"
        rows.append(row)

    for model_name, res in results_unsw.items():
        row = {"Model": model_name, "Dataset": "UNSW-NB15"}
        for m in metric_keys:
            row[m] = f"{res[m]}%"
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "Table3_Model_Performance_Comparison.csv")
    df.to_csv(path, index=False)
    print(f"    Saved: {path}")
    return df


def save_table4(model, X_test, y_test, dataset_name, out_dir):
    """Table 4: Per-class detection performance (proposed model)."""
    print(f"[5/6] Generating Table 4 — Per-Class Detection ({dataset_name})...")
    X_3d = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    y_prob = model.predict(X_3d, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(y_test, y_pred,
                                    target_names=["Normal", "Attack"],
                                    output_dict=True)
    df = pd.DataFrame(report).transpose().round(4)
    path = os.path.join(out_dir,
                        f"Table4_PerClass_Detection_{dataset_name}.csv")
    df.to_csv(path)
    print(f"    Saved: {path}")
    return df


def save_table5(results_cicids, results_unsw, train_times_cicids,
                train_times_unsw, out_dir):
    """Table 5: Computational cost comparison."""
    print("[5/6] Generating Table 5 — Computational Cost...")
    rows = []
    for model_name in results_cicids:
        rows.append({
            "Model"                  : model_name,
            "Train Time CICIDS (s)"  : train_times_cicids.get(model_name, "—"),
            "Train Time UNSW (s)"    : train_times_unsw.get(model_name, "—"),
            "Inference CICIDS (ms)"  : results_cicids[model_name]["Inf_ms"],
            "Inference UNSW (ms)"    : results_unsw[model_name]["Inf_ms"],
        })
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "Table5_Computational_Cost.csv")
    df.to_csv(path, index=False)
    print(f"    Saved: {path}")
    return df


def plot_roc_curves(results_cicids, results_unsw, y_test_cicids,
                    y_test_unsw, out_dir):
    """Figure 1: ROC curves for all models on both datasets."""
    print("[5/6] Generating Figure 1 — ROC Curves...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Figure 1: ROC Curves — All Models", fontsize=14, fontweight='bold')

    for ax, results, y_test, title in zip(
        axes,
        [results_cicids, results_unsw],
        [y_test_cicids, y_test_unsw],
        ["(a) CICIDS2017", "(b) UNSW-NB15"]
    ):
        color_list = [COLORS["proposed"], COLORS["rf"],
                      COLORS["svm"], COLORS["lstm"]]
        for (name, res), color in zip(results.items(), color_list):
            fpr_arr, tpr_arr, _ = roc_curve(y_test, res["y_prob"])
            ax.plot(fpr_arr, tpr_arr, color=color, lw=2,
                    label=f"{name} (AUC={res['AUC-ROC']}%)")
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label="Random Classifier")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "Figure1_ROC_Curves.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def plot_confusion_matrices(model_cicids, model_unsw,
                             X_test_cicids, y_test_cicids,
                             X_test_unsw, y_test_unsw, out_dir):
    """Figure 2: Confusion matrices for the proposed model."""
    print("[5/6] Generating Figure 2 — Confusion Matrices...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Figure 2: Confusion Matrix — Proposed CNN-LSTM Model",
                 fontsize=14, fontweight='bold')

    for ax, model, X_t, y_t, title in zip(
        axes,
        [model_cicids, model_unsw],
        [X_test_cicids, X_test_unsw],
        [y_test_cicids, y_test_unsw],
        ["(a) CICIDS2017", "(b) UNSW-NB15"]
    ):
        X_3d = X_t.reshape((X_t.shape[0], X_t.shape[1], 1))
        y_pred = (model.predict(X_3d, verbose=0).flatten() >= 0.5).astype(int)
        cm = confusion_matrix(y_t, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=["Normal", "Attack"],
                    yticklabels=["Normal", "Attack"])
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_title(title, fontsize=12)

        tn, fp, fn, tp = cm.ravel()
        ax.set_xlabel(
            f"Predicted Label\nTP={tp} | FP={fp} | FN={fn} | TN={tn}", fontsize=10
        )

    plt.tight_layout()
    path = os.path.join(out_dir, "Figure2_Confusion_Matrices.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def plot_training_curves(histories_cicids, histories_unsw, out_dir):
    """Figure 3: Training & validation loss/accuracy curves (deep models)."""
    print("[5/6] Generating Figure 3 — Training Curves...")
    deep_models = ["LSTM", "CNN-LSTM\n(Proposed)"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Figure 3: Training and Validation Curves — Deep Learning Models",
                 fontsize=13, fontweight='bold')

    for col_offset, (histories, ds_name) in enumerate(
        [(histories_cicids, "CICIDS2017"), (histories_unsw, "UNSW-NB15")]
    ):
        for row, model_key in enumerate(deep_models):
            if model_key not in histories:
                continue
            hist = histories[model_key].history
            epochs_ran = range(1, len(hist['loss']) + 1)
            col_loss = col_offset * 2
            col_acc  = col_offset * 2 + 1

            # Loss plot
            axes[row][col_loss].plot(epochs_ran, hist['loss'],
                                      color=COLORS["proposed"], label='Train Loss')
            axes[row][col_loss].plot(epochs_ran, hist.get('val_loss', []),
                                      color=COLORS["rf"], linestyle='--',
                                      label='Val Loss')
            axes[row][col_loss].set_title(
                f"{model_key.replace(chr(10), ' ')} — Loss ({ds_name})", fontsize=10)
            axes[row][col_loss].set_xlabel("Epoch")
            axes[row][col_loss].set_ylabel("Loss")
            axes[row][col_loss].legend(fontsize=8)
            axes[row][col_loss].grid(alpha=0.3)

            # Accuracy plot
            axes[row][col_acc].plot(epochs_ran, hist['accuracy'],
                                     color=COLORS["proposed"], label='Train Acc')
            axes[row][col_acc].plot(epochs_ran, hist.get('val_accuracy', []),
                                     color=COLORS["rf"], linestyle='--',
                                     label='Val Acc')
            axes[row][col_acc].set_title(
                f"{model_key.replace(chr(10), ' ')} — Accuracy ({ds_name})", fontsize=10)
            axes[row][col_acc].set_xlabel("Epoch")
            axes[row][col_acc].set_ylabel("Accuracy")
            axes[row][col_acc].legend(fontsize=8)
            axes[row][col_acc].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "Figure3_Training_Curves.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def plot_f1_comparison_bar(results_cicids, results_unsw, out_dir):
    """Bonus Figure 4: F1-Score bar chart comparison."""
    print("[5/6] Generating Figure 4 — F1 Comparison Bar Chart...")
    model_names = list(results_cicids.keys())
    f1_cicids   = [results_cicids[m]["F1-Score"] for m in model_names]
    f1_unsw     = [results_unsw[m]["F1-Score"] for m in model_names]
    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, f1_cicids, width,
                    label='CICIDS2017', color=COLORS["proposed"], alpha=0.85)
    bars2 = ax.bar(x + width/2, f1_unsw, width,
                    label='UNSW-NB15', color=COLORS["rf"], alpha=0.85)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("F1-Score (%)", fontsize=12)
    ax.set_title("Figure 4: F1-Score Comparison Across Models and Datasets",
                  fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('\n', ' ') for m in model_names], fontsize=10)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        ax.annotate(f'{bar.get_height():.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "Figure4_F1_Comparison.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


# =============================================================================
# SECTION 6 — FINAL METRICS LOG
# =============================================================================

def save_metrics_log(results_cicids, results_unsw,
                     train_times_cicids, train_times_unsw, out_dir):
    """Save a human-readable metrics log for paper writing."""
    path = os.path.join(out_dir, "METRICS_LOG_FOR_PAPER.txt")
    with open(path, 'w') as f:
        f.write("=" * 65 + "\n")
        f.write("  RESULTS LOG — AI-Based Cyber Threat Detection System\n")
        f.write("  Use these numbers to fill your paper sections\n")
        f.write("=" * 65 + "\n\n")

        for ds_name, results, times in [
            ("CICIDS2017",  results_cicids, train_times_cicids),
            ("UNSW-NB15",   results_unsw,   train_times_unsw)
        ]:
            f.write(f"\n{'─'*50}\n")
            f.write(f"  Dataset: {ds_name}\n")
            f.write(f"{'─'*50}\n")
            for model, res in results.items():
                model_clean = model.replace('\n', ' ')
                f.write(f"\n  Model : {model_clean}\n")
                f.write(f"    Accuracy  : {res['Accuracy']}%\n")
                f.write(f"    Precision : {res['Precision']}%\n")
                f.write(f"    Recall    : {res['Recall']}%\n")
                f.write(f"    F1-Score  : {res['F1-Score']}%\n")
                f.write(f"    AUC-ROC   : {res['AUC-ROC']}%\n")
                f.write(f"    FPR       : {res['FPR']}%\n")
                f.write(f"    Inf Time  : {res['Inf_ms']} ms/sample\n")
                f.write(f"    Train Time: {times.get(model, times.get(model_clean, '—'))}s\n")

        f.write("\n\n" + "=" * 65 + "\n")
        f.write("  OUTPUT FILES\n")
        f.write("=" * 65 + "\n")
        f.write("  Table 3 → Table3_Model_Performance_Comparison.csv\n")
        f.write("  Table 4 → Table4_PerClass_Detection_CICIDS2017.csv\n")
        f.write("            Table4_PerClass_Detection_UNSW-NB15.csv\n")
        f.write("  Table 5 → Table5_Computational_Cost.csv\n")
        f.write("  Figure 1 → Figure1_ROC_Curves.png\n")
        f.write("  Figure 2 → Figure2_Confusion_Matrices.png\n")
        f.write("  Figure 3 → Figure3_Training_Curves.png\n")
        f.write("  Figure 4 → Figure4_F1_Comparison.png\n")

    print(f"\n    Metrics log saved: {path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("\n" + "=" * 65)
    print("  AI-Based Cyber Threat Detection — Experiment Pipeline")
    print("=" * 65)

    # ── Load & preprocess CICIDS2017 ─────────────────────────────
    df_cicids      = load_cicids2017(DATA_CICIDS)
    X_c, y_c, _    = preprocess_cicids(df_cicids)
    splits_c       = split_and_scale(X_c, y_c)
    X_tr_c, X_v_c, X_te_c, y_tr_c, y_v_c, y_te_c = splits_c

    # ── Load & preprocess UNSW-NB15 ──────────────────────────────
    df_unsw        = load_unsw_nb15(DATA_UNSW)
    X_u, y_u, _    = preprocess_unsw(df_unsw)
    splits_u       = split_and_scale(X_u, y_u)
    X_tr_u, X_v_u, X_te_u, y_tr_u, y_v_u, y_te_u = splits_u

    print(f"\n[3/6] Data Summary:")
    print(f"  CICIDS2017 — Train: {X_tr_c.shape} | Val: {X_v_c.shape} | Test: {X_te_c.shape}")
    print(f"  UNSW-NB15  — Train: {X_tr_u.shape} | Val: {X_v_u.shape} | Test: {X_te_u.shape}")

    # ── Run experiments ──────────────────────────────────────────
    print("\n[4/6] Training models on both datasets...")
    (res_c, hist_c, times_c,
     model_c, X_te_c, y_te_c) = run_experiment(
        "CICIDS2017", X_tr_c, X_v_c, X_te_c, y_tr_c, y_v_c, y_te_c
    )
    (res_u, hist_u, times_u,
     model_u, X_te_u, y_te_u) = run_experiment(
        "UNSW-NB15", X_tr_u, X_v_u, X_te_u, y_tr_u, y_v_u, y_te_u
    )

    # ── Generate all outputs ──────────────────────────────────────
    print("\n[5/6] Generating tables and figures...")
    save_table3(res_c, res_u, RESULTS_DIR)
    save_table4(model_c, X_te_c, y_te_c, "CICIDS2017", RESULTS_DIR)
    save_table4(model_u, X_te_u, y_te_u, "UNSW-NB15",  RESULTS_DIR)
    save_table5(res_c, res_u, times_c, times_u, RESULTS_DIR)
    plot_roc_curves(res_c, res_u, y_te_c, y_te_u, RESULTS_DIR)
    plot_confusion_matrices(model_c, model_u,
                             X_te_c, y_te_c, X_te_u, y_te_u, RESULTS_DIR)
    plot_training_curves(hist_c, hist_u, RESULTS_DIR)
    plot_f1_comparison_bar(res_c, res_u, RESULTS_DIR)
    save_metrics_log(res_c, res_u, times_c, times_u, RESULTS_DIR)

    # ── Done ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ALL DONE — check the 'results/' folder")
    print("=" * 65)
    print(f"\n  Files generated:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(f"    ✓ results/{f}")
    print("\n  Open METRICS_LOG_FOR_PAPER.txt first — it has")
    print("  all numbers ready to paste into your paper.\n")


if __name__ == "__main__":
    main()
