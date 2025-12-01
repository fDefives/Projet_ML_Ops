#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_ml_flow_clean.py

Pipeline de training + Hyperopt + MLflow.
Principale modification MLflow : un seul run MLflow par essai Hyperopt.
Tous les folds/epochs/métriques et le meilleur modèle du trial sont logués
dans ce même run pour une lecture simple dans l'UI MLflow.
"""

import os
import time
import json
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

import torchvision.transforms as T

# ---------------- CONFIG ----------------
RAW_IMAGES_DIR = "data/lot1_images"
METADATA_DIR = "data/lot1_images"
RESIZED_IMAGES_DIR = "data/lot1_resized"
LABELS_CSV = "data/lot1_labels.csv"

BEARD_MAP_CSV = "beard.csv"
MUSTACHE_MAP_CSV = "mustache.csv"
GLASSES_MAP_CSV = "glasses.csv"
HAIR_COLOR_MAP_CSV = "hair_color.csv"
HAIR_MAP_CSV = "hair.csv"

TARGET_SIZE = (64, 64)
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 2
N_SPLITS = 5
SEED = 42

# ---------------- HYPEROPT CONFIG ----------------
HYPEROPT_MAX_EVALS = 20  # nombre d'essais

space = {
    "lr": hp.loguniform("lr", np.log(1e-4), np.log(5e-3)),
    "batch_size": hp.choice("batch_size", [64, 128, 256]),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MLFLOW_EXPERIMENT_NAME = "faces_multitask_kfold_v2_single_run"
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------- PREPROC ----------------
def segment_and_resize_images(input_folder, output_folder, target_size=(64, 64)):
    os.makedirs(output_folder, exist_ok=True)
    t0 = time.time()
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(".png"):
            continue
        img = Image.open(os.path.join(input_folder, filename)).convert("RGB")
        gray = img.convert("L")
        arr = np.array(gray)
        mask = arr < 250
        if not mask.any():
            cropped = img
        else:
            ys, xs = np.where(mask)
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            cropped = img.crop((x_min, y_min, x_max + 1, y_max + 1))
        resized = cropped.resize(target_size)
        resized.save(os.path.join(output_folder, filename))
    t1 = time.time()
    print(f"[PREPROC] images processed in {t1 - t0:.2f}s")

# ---------------- BUILD LABELS ----------------
def build_labels_table(metadata_dir,
                       beard_map_csv, mustache_map_csv, glasses_map_csv,
                       hair_color_map_csv, hair_map_csv,
                       out_csv):
    t0 = time.time()
    beard = pd.read_csv(beard_map_csv, sep=";")
    mustache = pd.read_csv(mustache_map_csv, sep=";")
    glasses = pd.read_csv(glasses_map_csv, sep=";")
    hair_color = pd.read_csv(hair_color_map_csv, sep=";")
    hair = pd.read_csv(hair_map_csv, sep=";")

    rows = []
    for file in os.listdir(metadata_dir):
        if not file.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(metadata_dir, file), header=None)
        df[0] = df[0].astype(str).str.replace('"', '').str.strip()

        def get_val(key):
            row = df[df[0] == key]
            return row.iloc[0, 1] if not row.empty else ""

        glasses_value_raw = get_val("glasses")
        glasses_value = 1 if glasses_value_raw in glasses.get("yes", pd.Series()).values else 0

        facial_hair_value = get_val("facial_hair")
        beard_value = 1 if facial_hair_value in beard.get("yes", pd.Series()).values else 0
        mustache_value = 1 if facial_hair_value in mustache.get("yes", pd.Series()).values else 0

        hair_color_value = get_val("hair_color")
        blond_value = 1 if hair_color_value in hair_color.get("blond", pd.Series()).values else 0
        light_brown_value = 1 if hair_color_value in hair_color.get("light_brown", pd.Series()).values else 0
        dark_brown_value = 1 if hair_color_value in hair_color.get("dark_brown", pd.Series()).values else 0
        redhead_value = 1 if hair_color_value in hair_color.get("redhead", pd.Series()).values else 0
        gray_blue_value = 1 if hair_color_value in hair_color.get("gray_blue", pd.Series()).values else 0

        hair_value = get_val("hair")
        long_value = 1 if hair_value in hair.get("long", pd.Series()).values else 0
        short_value = 1 if hair_value in hair.get("short", pd.Series()).values else 0
        bald_value = 1 if hair_value in hair.get("bald", pd.Series()).values else 0

        basename = os.path.splitext(file)[0]
        image_filename = f"{basename}.png"

        rows.append({
            "filename": image_filename,
            "glasses": glasses_value,
            "beard": beard_value,
            "mustache": mustache_value,
            "blond": blond_value,
            "light_brown": light_brown_value,
            "dark_brown": dark_brown_value,
            "redhead": redhead_value,
            "gray_blue": gray_blue_value,
            "long": long_value,
            "short": short_value,
            "bald": bald_value,
        })

    labels_df = pd.DataFrame(rows)
    labels_df.to_csv(out_csv, index=False)
    t1 = time.time()
    print(f"[LABELS] saved {out_csv} with {len(labels_df)} rows in {t1 - t0:.2f}s")
    return labels_df

# ---------------- DATASET ----------------
class FaceDataset(Dataset):
    def __init__(self, labels_csv, image_dir, transform=None):
        self.df = pd.read_csv(labels_csv)
        self.image_dir = image_dir
        self.transform = transform
        expected_cols = [
            "filename", "glasses", "beard", "mustache",
            "blond", "light_brown", "dark_brown", "redhead", "gray_blue",
            "long", "short", "bald"
        ]
        for c in expected_cols:
            if c not in self.df.columns:
                raise ValueError(f"Missing column in {labels_csv}: {c}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            arr = np.array(image, dtype=np.float32) / 255.0
            image = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        glasses = torch.tensor(row["glasses"], dtype=torch.float32)
        beard = torch.tensor(row["beard"], dtype=torch.float32)
        mustache = torch.tensor(row["mustache"], dtype=torch.float32)

        color_vector = np.array([
            row["blond"],
            row["light_brown"],
            row["redhead"],
            row["dark_brown"],
            row["gray_blue"],
        ], dtype=np.float32)
        color_class = int(color_vector.argmax())

        hair_vector = np.array([
            row["bald"],
            row["short"],
            row["long"],
        ], dtype=np.float32)
        hair_class = int(hair_vector.argmax())

        return {
            "image": image,
            "glasses": glasses,
            "beard": beard,
            "mustache": mustache,
            "color": torch.tensor(color_class, dtype=torch.long),
            "hair": torch.tensor(hair_class, dtype=torch.long),
        }

# ---------------- MODEL (exemple) ----------------
class CNNMultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten_dim = 128 * (TARGET_SIZE[0] // 8) * (TARGET_SIZE[1] // 8)
        self.shared_fc = nn.Sequential(nn.Linear(self.flatten_dim, 256), nn.ReLU())

        self.fc_glasses = nn.Linear(256, 1)
        self.fc_beard = nn.Linear(256, 1)
        self.fc_mustache = nn.Linear(256, 1)
        self.fc_color = nn.Linear(256, 5)
        self.fc_hair = nn.Linear(256, 3)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.shared_fc(x)
        return (
            self.fc_glasses(x),
            self.fc_beard(x),
            self.fc_mustache(x),
            self.fc_color(x),
            self.fc_hair(x),
        )

# ---------------- TRAIN / EVAL ----------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        glasses = batch["glasses"].to(device)
        beard = batch["beard"].to(device)
        mustache = batch["mustache"].to(device)
        color = batch["color"].to(device)
        hair = batch["hair"].to(device)

        optimizer.zero_grad()
        g_logits, b_logits, m_logits, c_logits, h_logits = model(images)

        loss = (F.binary_cross_entropy_with_logits(g_logits.squeeze(), glasses)
                + F.binary_cross_entropy_with_logits(b_logits.squeeze(), beard)
                + F.binary_cross_entropy_with_logits(m_logits.squeeze(), mustache)
                + F.cross_entropy(c_logits, color)
                + F.cross_entropy(h_logits, hair))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    true_glasses = []; pred_glasses = []
    true_beard = []; pred_beard = []
    true_must = []; pred_must = []
    true_color = []; pred_color = []
    true_hair = []; pred_hair = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            glasses = batch["glasses"].to(device)
            beard = batch["beard"].to(device)
            mustache = batch["mustache"].to(device)
            color = batch["color"].to(device)
            hair = batch["hair"].to(device)

            g_logits, b_logits, m_logits, c_logits, h_logits = model(images)

            loss = (F.binary_cross_entropy_with_logits(g_logits.squeeze(), glasses)
                    + F.binary_cross_entropy_with_logits(b_logits.squeeze(), beard)
                    + F.binary_cross_entropy_with_logits(m_logits.squeeze(), mustache)
                    + F.cross_entropy(c_logits, color)
                    + F.cross_entropy(h_logits, hair))

            total_loss += loss.item()
            n_batches += 1

            g_pred = (torch.sigmoid(g_logits.squeeze()) > 0.5).long().cpu().numpy()
            b_pred = (torch.sigmoid(b_logits.squeeze()) > 0.5).long().cpu().numpy()
            m_pred = (torch.sigmoid(m_logits.squeeze()) > 0.5).long().cpu().numpy()
            c_pred = torch.argmax(c_logits, dim=1).cpu().numpy()
            h_pred = torch.argmax(h_logits, dim=1).cpu().numpy()

            g_true = glasses.long().cpu().numpy()
            b_true = beard.long().cpu().numpy()
            m_true = mustache.long().cpu().numpy()
            c_true = color.cpu().numpy()
            h_true = hair.cpu().numpy()

            pred_glasses.extend(g_pred.tolist())
            pred_beard.extend(b_pred.tolist())
            pred_must.extend(m_pred.tolist())
            pred_color.extend(c_pred.tolist())
            pred_hair.extend(h_pred.tolist())

            true_glasses.extend(g_true.tolist())
            true_beard.extend(b_true.tolist())
            true_must.extend(m_true.tolist())
            true_color.extend(c_true.tolist())
            true_hair.extend(h_true.tolist())

    avg_loss = total_loss / max(1, n_batches)

    def safe_acc(t, p):
        if len(t) == 0:
            return 0.0
        return float((np.array(t) == np.array(p)).mean())

    acc_gl = safe_acc(true_glasses, pred_glasses)
    acc_be = safe_acc(true_beard, pred_beard)
    acc_mu = safe_acc(true_must, pred_must)
    acc_co = safe_acc(true_color, pred_color)
    acc_ha = safe_acc(true_hair, pred_hair)

    try:
        f1_gl = f1_score(true_glasses, pred_glasses, zero_division=0)
        f1_be = f1_score(true_beard, pred_beard, zero_division=0)
        f1_mu = f1_score(true_must, pred_must, zero_division=0)
        f1_co = f1_score(true_color, pred_color, average="macro", zero_division=0)
        f1_ha = f1_score(true_hair, pred_hair, average="macro", zero_division=0)
    except Exception:
        f1_gl = f1_be = f1_mu = f1_co = f1_ha = 0.0

    metrics = {
        "val_loss": avg_loss,
        "acc_glasses": acc_gl,
        "acc_beard": acc_be,
        "acc_mustache": acc_mu,
        "acc_color": acc_co,
        "acc_hair": acc_ha,
        "f1_glasses": float(f1_gl),
        "f1_beard": float(f1_be),
        "f1_mustache": float(f1_mu),
        "f1_color_macro": float(f1_co),
        "f1_hair_macro": float(f1_ha),
    }
    return metrics

# ---------------- K-FOLD TRAIN (no nested MLflow runs) ----------------
def train_kfold(dataset, n_splits, epochs, batch_size, lr, device):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    best_mean_acc = -1.0
    best_state = None
    best_info = {}

    # On suppose qu'un run MLflow est déjà ouvert au niveau caller (hyperopt_objective)
    # Log quelques paramètres utiles
    mlflow.log_param("n_splits", n_splits)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("lr", lr)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(dataset)))):
        print(f"\n===== FOLD {fold} =====")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        workers = min(4, (os.cpu_count() or 1))
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=(workers > 0),
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=(workers > 0),
        )

        model = CNNMultiTask().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Pour stocker les metrics du dernier epoch du fold (utilisé pour mean_acc)
        last_val_metrics = None

        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)
            last_val_metrics = val_metrics

            print(f"Fold {fold} | Epoch {epoch} | train_loss={train_loss:.4f} | val_loss={val_metrics['val_loss']:.4f}")

            # log metrics avec préfixe fold{fold}_
            mlflow.log_metric(f"fold{fold}_train_loss", train_loss, step=epoch)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"fold{fold}_{k}", v, step=epoch)

        # mean acc pour comparaison du fold (on utilise le dernier val_metrics du fold)
        mean_acc = (
            last_val_metrics["acc_glasses"]
            + last_val_metrics["acc_beard"]
            + last_val_metrics["acc_mustache"]
            + last_val_metrics["acc_color"]
            + last_val_metrics["acc_hair"]
        ) / 5.0
        mlflow.log_metric(f"fold{fold}_mean_acc", mean_acc)

        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            best_state = model.state_dict()
            # stocke info sommaire
            best_info = {
                "fold": fold,
                "mean_acc": float(mean_acc)
            }

    # résumé K-fold
    mlflow.log_metric("best_mean_acc", best_mean_acc)
    if best_info:
        mlflow.set_tag("best_fold", best_info["fold"])

    print(f"\n[TRAIN] Finished K-Fold. best_mean_acc={best_mean_acc}, best_fold={best_info.get('fold')}")
    return best_state, best_info

# ---------------- HYPEROPT OBJECTIVE (opens a single MLflow run per trial) ----------------
def hyperopt_objective(params, dataset):
    lr = float(params["lr"])
    batch_size = int(params["batch_size"])
    epochs = EPOCHS

    # Ouvre un seul run MLflow pour cet essai Hyperopt (trial)
    run_name = f"hyperopt_lr{lr:.2e}_bs{batch_size}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("n_splits", N_SPLITS)
        mlflow.log_param("seed", SEED)
        mlflow.log_param("target_size", TARGET_SIZE)

        print(f"\n[HYPEROPT] Trial avec lr={lr:.5f}, batch_size={batch_size}, epochs={epochs}")

        # Entraînement K-Fold — tout log dans le même run
        best_state, best_info = train_kfold(
            dataset=dataset,
            n_splits=N_SPLITS,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=DEVICE
        )

        mean_acc = best_info.get("mean_acc", 0.0)
        loss = 1.0 - float(mean_acc)

        # Log métrique objective (optimisation)
        mlflow.log_metric("objective_loss", loss)
        mlflow.log_metric("objective_mean_acc", float(mean_acc))

        # Sauvegarde du meilleur state_dict du trial en artifact dans ce même run
        if best_state is not None:
            model_path = f"best_model_trial_lr{lr:.2e}_bs{batch_size}.pth"
            torch.save(best_state, model_path)
            try:
                mlflow.log_artifact(model_path, artifact_path="best_model")
                # Optionnel : log du modèle en tant que pytorch model (avec wrapper si nécessaire)
                try:
                    # wrapper pour concat outputs -> unique tensor (utile pour signature)
                    class ModelWrapper(nn.Module):
                        def __init__(self, base):
                            super().__init__()
                            self.base = base

                        def forward(self, x):
                            outs = self.base(x)
                            if isinstance(outs, (tuple, list)):
                                parts = [o.view(o.size(0), -1) for o in outs]
                                return torch.cat(parts, dim=1)
                            return outs

                    # on ne peut pas reconstruire le modèle complet facilement ici sans l'instance,
                    # donc on évite mlflow.pytorch.log_model sauf si on a le module complet en cpu.
                    # On sauvegarde le state_dict comme artifact (déjà fait).
                except Exception:
                    pass
            except Exception as e:
                print("[MLFLOW] Warning: failed to log artifact:", e)

        # Save a small JSON summary artifact for easy reading
        summary = {
            "trial_params": params,
            "best_info": best_info,
            "objective_loss": float(loss),
        }
        with open("trial_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        try:
            mlflow.log_artifact("trial_summary.json", artifact_path="summary")
        except Exception:
            pass

    # Hyperopt wants a dict with 'loss' and 'status'
    return {
        "loss": float(loss),
        "status": STATUS_OK,
        "mean_acc": float(mean_acc),
        "params": params,
    }

# ---------------- MAIN ----------------
def main():
    print("[DEVICE] torch.cuda.is_available():", torch.cuda.is_available())
    print("[DEVICE] DEVICE variable:", DEVICE)
    if torch.cuda.is_available():
        try:
            print("[DEVICE] cuda count:", torch.cuda.device_count())
            print("[DEVICE] cuda name:", torch.cuda.get_device_name(0))
        except Exception as e:
            print("[DEVICE] cuda info error:", e)

    # 1) preproc images (skip if exists)
    resized_exists = os.path.isdir(RESIZED_IMAGES_DIR) and any(
        f.lower().endswith(".png") for f in os.listdir(RESIZED_IMAGES_DIR)
    )
    if not resized_exists:
        print("Running image preproc...")
        segment_and_resize_images(RAW_IMAGES_DIR, RESIZED_IMAGES_DIR, TARGET_SIZE)
    else:
        print(f"[PREPROC] {RESIZED_IMAGES_DIR} already exists -> skip")

    # 2) build labels if missing
    labels_exists = os.path.exists(LABELS_CSV) and os.path.getsize(LABELS_CSV) > 0
    if not labels_exists:
        print("Building labels csv...")
        build_labels_table(
            METADATA_DIR,
            BEARD_MAP_CSV,
            MUSTACHE_MAP_CSV,
            GLASSES_MAP_CSV,
            HAIR_COLOR_MAP_CSV,
            HAIR_MAP_CSV,
            LABELS_CSV
        )
    else:
        print(f"[LABELS] {LABELS_CSV} exists -> skip")

    # 3) dataset
    if not os.path.exists(LABELS_CSV):
        raise RuntimeError(f"Missing labels file: {LABELS_CSV}")
    if not os.path.isdir(RESIZED_IMAGES_DIR):
        raise RuntimeError(f"Missing resized dir: {RESIZED_IMAGES_DIR}")

    transform = T.Compose([
        T.Resize(TARGET_SIZE),
        T.ToTensor()
    ])

    dataset = FaceDataset(LABELS_CSV, RESIZED_IMAGES_DIR, transform=transform)
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) < 2:
        raise RuntimeError("Dataset too small for k-fold")

    # -------- HYPEROPT: recherche de hyperparamètres --------
    trials = Trials()

    # wrapper pour passer dataset à hyperopt_objective
    def objective_with_dataset(params):
        return hyperopt_objective(params, dataset)

    best = fmin(
        fn=objective_with_dataset,
        space=space,
        algo=tpe.suggest,
        max_evals=HYPEROPT_MAX_EVALS,
        trials=trials,
        rstate=np.random.default_rng(SEED)
    )

    print("\n[HYPEROPT] Meilleurs hyperparamètres trouvés :", best)

    # Récupérer la meilleure trial (la plus petite loss)
    best_trial = sorted(trials.results, key=lambda r: r["loss"])[0]
    print(f"[HYPEROPT] Best mean_acc={best_trial.get('mean_acc', 0.0):.4f} avec params={best_trial.get('params')}")

    # On peut réentraîner final si besoin et loguer dans un run séparé
    best_lr = float(best_trial["params"]["lr"])
    best_batch_size = int(best_trial["params"]["batch_size"])

    print("[FINAL TRAIN] Optionnel: réentraînement final avec les meilleurs hyperparamètres...")
    final_state, final_info = train_kfold(
        dataset=dataset,
        n_splits=N_SPLITS,
        epochs=EPOCHS,
        batch_size=best_batch_size,
        lr=best_lr,
        device=DEVICE
    )

    if final_state is not None:
        model_path = "cnn_multitask_best_final.pth"
        torch.save(final_state, model_path)
        print(f"[SAVE] saved best final state to {model_path}")

        # Log final model in a dedicated run for clarity
        with mlflow.start_run(run_name="final_model_retrain"):
            mlflow.log_param("final_lr", best_lr)
            mlflow.log_param("final_batch_size", best_batch_size)
            mlflow.log_metric("final_best_mean_acc", final_info.get("mean_acc", 0.0))
            try:
                mlflow.log_artifact(model_path, artifact_path="best_model_final")
                print("[MLFLOW] best final model artifact logged")
            except Exception as e:
                print("[MLFLOW] Warning: failed to log final artifact:", e)
    else:
        print("[SAVE] No best model found from final training")

if __name__ == "__main__":
    main()
