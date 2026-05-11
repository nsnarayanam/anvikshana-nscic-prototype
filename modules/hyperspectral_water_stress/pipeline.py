"""
Groundnut Water Stress Classification — Hyperspectral Pipeline
================================================================
End-to-end pipeline: load → preprocess → train (RF / SVM / 1D-CNN)
→ evaluate → save figures + metrics.

Author:  Narasimha Sharma Narayanam, Aganitha Space Technologies
Dataset: IIT-H TiHAN — UC-HSI Crop Variety (Groundnut Water Stress)

Usage:
    python groundnut_pipeline.py --data_dir /path/to/Crop_dataset_groundnut

Inputs expected in --data_dir:
    X_GN_31Dec.npy   shape: (N, 300), float32/64, reflectance
    y_GN_31Dec.npy   shape: (N,),     labels in {1, 2}

Outputs written to --out_dir:
    groundnut_water_stress_cnn.pt           trained CNN weights
    groundnut_water_stress_results.json     all metrics
    fig_mean_spectra.png
    fig_difference_spectrum.png
    fig_rf_importance.png
"""

from __future__ import annotations
import argparse, json, os, time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)


# -------------------------------------------------------------- Model

class SpectralCNN(nn.Module):
    """1D-CNN over hyperspectral pixel signatures."""

    def __init__(self, n_bands: int = 300, n_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# -------------------------------------------------------------- Utils

def metrics_block(name: str, y_true, y_pred, y_proba=None) -> dict:
    """Compute & print standard metrics, return as dict."""
    out = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
    }
    if y_proba is not None:
        out['roc_auc'] = float(roc_auc_score(y_true, y_proba))
    out['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    print(f"\n--- {name} ---")
    print(f"Accuracy : {out['accuracy']:.4f}")
    print(f"F1 macro : {out['f1_macro']:.4f}")
    if 'roc_auc' in out:
        print(f"ROC-AUC  : {out['roc_auc']:.4f}")
    print("Confusion matrix:", out['confusion_matrix'])
    print(classification_report(y_true, y_pred, digits=4))
    return out


def evaluate_torch(model, loader, device):
    model.eval()
    preds, probs, ys = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            preds.append(out.argmax(1).cpu().numpy())
            probs.append(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
            ys.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(probs), np.concatenate(ys)


# -------------------------------------------------------------- Main

def run(data_dir: Path, out_dir: Path, n_epochs: int = 40,
        seed: int = 42, x_filename: str = 'X_GN_31Dec.npy',
        y_filename: str = 'y_GN_31Dec.npy') -> dict:

    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed); torch.manual_seed(seed)

    # --- Load
    X = np.load(data_dir / x_filename).astype(np.float32)
    y_raw = np.load(data_dir / y_filename)
    # remap from {1, 2} to {0, 1} if needed
    classes = sorted(np.unique(y_raw).tolist())
    if classes != [0, 1]:
        y = (y_raw - classes[0]).astype(np.int64)
    else:
        y = y_raw.astype(np.int64)

    print(f"Loaded X={X.shape} y_counts={np.bincount(y).tolist()}")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN/Inf — clean the data before training.")

    # --- Stratified 70/15/15 split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=seed)

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s, X_test_s = scaler.transform(X_train), \
                                   scaler.transform(X_val), \
                                   scaler.transform(X_test)

    results = {'config': {'seed': seed, 'n_epochs': n_epochs,
                          'n_bands': int(X.shape[1]),
                          'splits': {'train': len(X_train),
                                     'val': len(X_val), 'test': len(X_test)}}}

    # --- Random Forest
    print("\n=== Random Forest ===")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                class_weight='balanced', random_state=seed)
    rf.fit(X_train_s, y_train)
    print(f"Trained in {time.time()-t0:.1f}s")
    results['rf'] = {
        'val':  metrics_block('VAL',  y_val,  rf.predict(X_val_s),
                              rf.predict_proba(X_val_s)[:, 1]),
        'test': metrics_block('TEST', y_test, rf.predict(X_test_s),
                              rf.predict_proba(X_test_s)[:, 1]),
        'top_bands_by_importance':
            np.argsort(rf.feature_importances_)[-15:][::-1].tolist(),
    }

    # --- SVM RBF
    print("\n=== SVM (RBF) ===")
    t0 = time.time()
    svm = SVC(C=10, gamma='scale', kernel='rbf',
              class_weight='balanced', probability=True, random_state=seed)
    svm.fit(X_train_s, y_train)
    print(f"Trained in {time.time()-t0:.1f}s")
    results['svm'] = {
        'val':  metrics_block('VAL',  y_val,  svm.predict(X_val_s),
                              svm.predict_proba(X_val_s)[:, 1]),
        'test': metrics_block('TEST', y_test, svm.predict(X_test_s),
                              svm.predict_proba(X_test_s)[:, 1]),
    }

    # --- 1D-CNN
    print("\n=== 1D-CNN ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    def make_loader(X, y, bs, shuffle):
        ds = TensorDataset(
            torch.tensor(X[:, None, :], dtype=torch.float32),
            torch.tensor(y, dtype=torch.long))
        return DataLoader(ds, batch_size=bs, shuffle=shuffle)

    tr_loader = make_loader(X_train_s, y_train, 128, True)
    va_loader = make_loader(X_val_s,   y_val,   256, False)
    te_loader = make_loader(X_test_s,  y_test,  256, False)

    model = SpectralCNN(n_bands=X.shape[1]).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit  = nn.CrossEntropyLoss()

    best_f1, best_state = 0.0, None
    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        sched.step()
        vp, _, vy = evaluate_torch(model, va_loader, device)
        f1v = f1_score(vy, vp, average='macro')
        if f1v > best_f1:
            best_f1 = f1v
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:2d}  val_F1={f1v:.4f}")
    print(f"Trained in {time.time()-t0:.1f}s; best val F1={best_f1:.4f}")

    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / 'groundnut_water_stress_cnn.pt')

    vp, vpr, vy = evaluate_torch(model, va_loader, device)
    tp, tpr, ty = evaluate_torch(model, te_loader, device)
    results['cnn'] = {
        'val':  metrics_block('VAL',  vy, vp, vpr),
        'test': metrics_block('TEST', ty, tp, tpr),
    }

    # --- Baselines: single-band + brightness + linear full spectrum
    print("\n=== Baselines (separability diagnostics) ===")
    baselines = {}
    for b in results['rf']['top_bands_by_importance'][:5]:
        acc = LogisticRegression(max_iter=2000).fit(
            X_train_s[:, [b]], y_train).score(X_test_s[:, [b]], y_test)
        baselines[f'logreg_band_{b}'] = float(acc)
        print(f"  logreg band {b:3d} alone: {acc:.4f}")
    mb_train = X_train.mean(axis=1, keepdims=True)
    mb_test  = X_test.mean(axis=1,  keepdims=True)
    baselines['logreg_brightness'] = float(LogisticRegression(max_iter=2000)
        .fit(mb_train, y_train).score(mb_test, y_test))
    baselines['logreg_full_spectrum'] = float(LogisticRegression(max_iter=2000, C=1.0)
        .fit(X_train_s, y_train).score(X_test_s, y_test))
    print(f"  logreg brightness only      : {baselines['logreg_brightness']:.4f}")
    print(f"  logreg full spectrum (linear): {baselines['logreg_full_spectrum']:.4f}")
    results['baselines'] = baselines

    # --- Diagnostic figures
    print("\n=== Saving figures ===")
    mean0 = X[y == 0].mean(axis=0); std0 = X[y == 0].std(axis=0)
    mean1 = X[y == 1].mean(axis=0); std1 = X[y == 1].std(axis=0)

    plt.figure(figsize=(12, 5))
    bands = np.arange(X.shape[1])
    plt.plot(mean0, label=f'Class 0 (n={(y==0).sum()})', linewidth=2)
    plt.fill_between(bands, mean0-std0, mean0+std0, alpha=0.2)
    plt.plot(mean1, label=f'Class 1 (n={(y==1).sum()})', linewidth=2)
    plt.fill_between(bands, mean1-std1, mean1+std1, alpha=0.2)
    plt.xlabel('Spectral band index'); plt.ylabel('Reflectance')
    plt.title('Mean spectral signature per class (±1 std)')
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(out_dir / 'fig_mean_spectra.png', dpi=200, bbox_inches='tight')
    plt.close()

    diff = mean0 - mean1
    se   = np.sqrt(std0**2/(y==0).sum() + std1**2/(y==1).sum())
    t    = diff / (se + 1e-12)
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(diff, color='tab:red'); ax1.axhline(0, color='k', linewidth=0.5)
    ax1.set_xlabel('Band index')
    ax1.set_ylabel('Reflectance difference (C0 − C1)', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red'); ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(np.abs(t), color='tab:purple', alpha=0.5)
    ax2.set_ylabel('|t-statistic|', color='tab:purple')
    ax2.tick_params(axis='y', labelcolor='tab:purple')
    plt.title('Where does the water-stress signal live?')
    plt.savefig(out_dir / 'fig_difference_spectrum.png', dpi=200, bbox_inches='tight')
    plt.close()

    imp = rf.feature_importances_
    top = np.argsort(imp)[-15:][::-1]
    plt.figure(figsize=(10, 4))
    plt.bar(range(15), imp[top])
    plt.xticks(range(15), top)
    plt.xlabel('Spectral band index'); plt.ylabel('Importance')
    plt.title('Random Forest — top 15 spectral bands for water stress')
    plt.grid(alpha=0.3, axis='y')
    plt.savefig(out_dir / 'fig_rf_importance.png', dpi=200, bbox_inches='tight')
    plt.close()

    # --- Persist results
    with open(out_dir / 'groundnut_water_stress_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nAll outputs written to {out_dir}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--out_dir',  type=Path, default=Path('./outputs'))
    parser.add_argument('--epochs',   type=int,  default=40)
    parser.add_argument('--seed',     type=int,  default=42)
    args = parser.parse_args()
    run(args.data_dir, args.out_dir, n_epochs=args.epochs, seed=args.seed)
