#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
script.py
Pipeline complet :
  1. Prétraitement images (segmentation simple + resize)
  2. Construction du CSV de labels à partir des fichiers .csv de description
  3. Dataset PyTorch
  4. Entraînement k-fold avec MLflow
  5. Sauvegarde du meilleur modèle (.pth) + log dans MLflow

⚠️ A adapter :
  - Les chemins des dossiers (images brutes, csv, etc.)
  - Le modèle CNNMultiTask (remplace l'exemple par le tien)
  - Le transform (normalisation, etc.)
"""

import os
import time
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim

from sklearn.model_selection import KFold

import mlflow
import mlflow.pytorch

# ---------------------------------------------------------------------
# 0. CONFIG GLOBALE
# ---------------------------------------------------------------------

# TODO: adapte ces chemins à ta structure de fichiers
RAW_IMAGES_DIR = "data/lot1_images"       # images d'entraînement brutes (PNG)
METADATA_DIR = "data/lot1_images"           # CSV de description par image (glasses, facial_hair, etc.)
RESIZED_IMAGES_DIR = "data/lot1_resized"      # où sauver les images recadrées + redimensionnées
LABELS_CSV = "data/lot1_labels.csv"           # CSV global des labels (une ligne par image)

# fichiers de mapping déjà existants chez toi
BEARD_MAP_CSV = "beard.csv"
MUSTACHE_MAP_CSV = "mustache.csv"
GLASSES_MAP_CSV = "glasses.csv"
HAIR_COLOR_MAP_CSV = "hair_color.csv"
HAIR_MAP_CSV = "hair.csv"

TARGET_SIZE = (64, 64)
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 5
N_SPLITS = 5
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MLflow
MLFLOW_EXPERIMENT_NAME = "faces_multitask_kfold"


# ---------------------------------------------------------------------
# 1. PRÉTRAITEMENT IMAGES : segmentation simple + resize
# ---------------------------------------------------------------------

def segment_and_resize_images(input_folder, output_folder, target_size=(64, 64)):
    """
    Segmentation simple (suppression du fond blanc) + resize.
    Reprend exactement ta logique de traitement.
    """
    os.makedirs(output_folder, exist_ok=True)

    time_start = time.time()

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(".png"):
            continue

        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGB")

        # niveaux de gris pour construire le masque
        gray = img.convert("L")
        arr = np.array(gray)

        # Masque : pixels non blancs (ou pas trop blancs)
        mask = arr < 250

        # Si tout est blanc, on ne recadre pas
        if not mask.any():
            cropped = img
        else:
            ys, xs = np.where(mask)
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            cropped = img.crop((x_min, y_min, x_max + 1, y_max + 1))

        img_resized = cropped.resize(target_size)
        img_resized.save(os.path.join(output_folder, filename))

    time_end = time.time()
    print(f"[PREPROC] Temps de traitement des images : {time_end - time_start:.2f} s")


# ---------------------------------------------------------------------
# 2. CONSTRUCTION DU CSV GLOBAL DE LABELS
# ---------------------------------------------------------------------

def build_labels_table(metadata_dir,
                       beard_map_csv, mustache_map_csv, glasses_map_csv,
                       hair_color_map_csv, hair_map_csv,
                       out_csv):
    """
    Parcourt tous les fichiers .csv de metadata dans metadata_dir
    et construit un CSV global (une ligne par image) :
    filename,glasses,beard,mustache,blond,light_brown,dark_brown,redhead,gray_blue,long,short,bald
    """
    time_start = time.time()

    # Chargement des map CSV
    beard = pd.read_csv(beard_map_csv, sep=";")
    mustache = pd.read_csv(mustache_map_csv, sep=";")
    glasses = pd.read_csv(glasses_map_csv, sep=";")
    hair_color = pd.read_csv(hair_color_map_csv, sep=";")
    hair = pd.read_csv(hair_map_csv, sep=";")

    rows = []

    for file in os.listdir(metadata_dir):
        if not file.endswith(".csv"):
            continue

        meta_path = os.path.join(metadata_dir, file)
        df = pd.read_csv(meta_path, header=None)
        df[0] = df[0].str.replace('"', '').str.strip()

        # --- glasses ---
        row_glasses = df[df[0] == "glasses"]
        glasses_value_raw = row_glasses.iloc[0, 1] if not row_glasses.empty else ""
        glasses_value = 1 if glasses_value_raw in glasses["yes"].values else 0

        # --- facial_hair -> beard + mustache ---
        row_fh = df[df[0] == "facial_hair"]
        facial_hair_value = row_fh.iloc[0, 1] if not row_fh.empty else ""

        beard_value = 1 if facial_hair_value in beard["yes"].values else 0
        mustache_value = 1 if facial_hair_value in mustache["yes"].values else 0

        # --- hair_color -> 5 one-hot ---
        row_hc = df[df[0] == "hair_color"]
        hair_color_value = row_hc.iloc[0, 1] if not row_hc.empty else ""

        blond_value = 1 if hair_color_value in hair_color["blond"].values else 0
        light_brown_value = 1 if hair_color_value in hair_color["light_brown"].values else 0
        dark_brown_value = 1 if hair_color_value in hair_color["dark_brown"].values else 0
        redhead_value = 1 if hair_color_value in hair_color["redhead"].values else 0
        gray_blue_value = 1 if hair_color_value in hair_color["gray_blue"].values else 0

        # --- hair -> long, short, bald ---
        row_h = df[df[0] == "hair"]
        hair_value = row_h.iloc[0, 1] if not row_h.empty else ""

        long_value = 1 if hair_value in hair["long"].values else 0
        short_value = 1 if hair_value in hair["short"].values else 0
        bald_value = 1 if hair_value in hair["bald"].values else 0

        # On suppose que l'image associée est <basename>.png
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
    time_end = time.time()
    print(f"[LABELS] CSV global sauvegardé dans {out_csv} ({len(labels_df)} lignes)")
    print(f"[LABELS] Temps de traitement des .csv : {time_end - time_start:.2f} s")

    return labels_df


# ---------------------------------------------------------------------
# 3. DATASET PYTORCH
# ---------------------------------------------------------------------

class FaceDataset(Dataset):
    """
    Dataset d'entraînement :
      - images dans RESIZED_IMAGES_DIR
      - labels dans LABELS_CSV

    Targets :
      - glasses, beard, mustache : 0/1 (float)
      - color_class : int [0..4] (blond, light_brown, redhead, dark_brown, gray_blue)
      - hair_class : int [0..2] (bald, short, long) -> ordre choisi ici
    """

    def __init__(self, labels_csv, image_dir, transform=None):
        self.df = pd.read_csv(labels_csv)
        self.image_dir = image_dir
        self.transform = transform

        # On vérifie que les colonnes nécessaires existent
        expected_cols = [
            "filename", "glasses", "beard", "mustache",
            "blond", "light_brown", "dark_brown", "redhead", "gray_blue",
            "long", "short", "bald"
        ]
        for c in expected_cols:
            if c not in self.df.columns:
                raise ValueError(f"Colonne manquante dans {labels_csv}: {c}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            # transform par défaut : tensor [0,1]
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0

        # Binaires
        glasses = torch.tensor(row["glasses"], dtype=torch.float32)
        beard = torch.tensor(row["beard"], dtype=torch.float32)
        mustache = torch.tensor(row["mustache"], dtype=torch.float32)

        # Multi-classes couleur
        color_vector = np.array([
            row["blond"],
            row["light_brown"],
            row["redhead"],
            row["dark_brown"],
            row["gray_blue"],
        ], dtype=np.float32)
        color_class = int(color_vector.argmax())

        # Multi-classes taille cheveux (ordre: bald, short, long)
        hair_vector = np.array([
            row["bald"],
            row["short"],
            row["long"],
        ], dtype=np.float32)
        hair_class = int(hair_vector.argmax())

        color = torch.tensor(color_class, dtype=torch.long)
        hair = torch.tensor(hair_class, dtype=torch.long)

        return {
            "image": image,
            "glasses": glasses,
            "beard": beard,
            "mustache": mustache,
            "color": color,
            "hair": hair,
        }


# ---------------------------------------------------------------------
# 4. MODELE CNN MULTI-TÂCHES (EXEMPLE)
# ---------------------------------------------------------------------

class CNNMultiTask(nn.Module):
    """
    Exemple de modèle multi-tâches.
    ⚠️ Remplace par TON architecture si tu en as déjà une.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 64x64 -> 64x64
            nn.ReLU(),
            nn.MaxPool2d(2),                # 32x32
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                # 16x16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                # 8x8
        )
        self.flatten_dim = 128 * 8 * 8

        self.shared_fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
        )

        # 3 têtes binaires (sigmoid ensuite)
        self.fc_glasses = nn.Linear(256, 1)
        self.fc_beard = nn.Linear(256, 1)
        self.fc_mustache = nn.Linear(256, 1)

        # couleur cheveux (5 classes)
        self.fc_color = nn.Linear(256, 5)

        # taille cheveux (3 classes)
        self.fc_hair = nn.Linear(256, 3)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.shared_fc(x)

        glasses_logits = self.fc_glasses(x)
        beard_logits = self.fc_beard(x)
        must_logits = self.fc_mustache(x)
        color_logits = self.fc_color(x)
        hair_logits = self.fc_hair(x)

        return glasses_logits, beard_logits, must_logits, color_logits, hair_logits


# ---------------------------------------------------------------------
# 5. BOUCLES D'ENTRAÎNEMENT / EVAL (K-FOLD)
# ---------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        glasses = batch["glasses"].to(device)
        beard = batch["beard"].to(device)
        mustache = batch["mustache"].to(device)
        color = batch["color"].to(device)
        hair = batch["hair"].to(device)

        optimizer.zero_grad()

        glasses_logits, beard_logits, must_logits, color_logits, hair_logits = model(images)

        loss_glasses = F.binary_cross_entropy_with_logits(glasses_logits.squeeze(), glasses)
        loss_beard = F.binary_cross_entropy_with_logits(beard_logits.squeeze(), beard)
        loss_must = F.binary_cross_entropy_with_logits(must_logits.squeeze(), mustache)
        loss_color = F.cross_entropy(color_logits, color)
        loss_hair = F.cross_entropy(hair_logits, hair)

        loss = loss_glasses + loss_beard + loss_must + loss_color + loss_hair
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    correct_glasses = correct_beard = correct_must = 0
    correct_color = correct_hair = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            glasses = batch["glasses"].to(device)
            beard = batch["beard"].to(device)
            mustache = batch["mustache"].to(device)
            color = batch["color"].to(device)
            hair = batch["hair"].to(device)

            glasses_logits, beard_logits, must_logits, color_logits, hair_logits = model(images)

            loss_glasses = F.binary_cross_entropy_with_logits(glasses_logits.squeeze(), glasses)
            loss_beard = F.binary_cross_entropy_with_logits(beard_logits.squeeze(), beard)
            loss_must = F.binary_cross_entropy_with_logits(must_logits.squeeze(), mustache)
            loss_color = F.cross_entropy(color_logits, color)
            loss_hair = F.cross_entropy(hair_logits, hair)

            loss = loss_glasses + loss_beard + loss_must + loss_color + loss_hair
            total_loss += loss.item()
            n_batches += 1

            batch_size = images.size(0)
            total_samples += batch_size

            pred_glasses = (torch.sigmoid(glasses_logits.squeeze()) > 0.5).long()
            pred_beard = (torch.sigmoid(beard_logits.squeeze()) > 0.5).long()
            pred_must = (torch.sigmoid(must_logits.squeeze()) > 0.5).long()
            pred_color = torch.argmax(color_logits, dim=1)
            pred_hair = torch.argmax(hair_logits, dim=1)

            correct_glasses += (pred_glasses == glasses.long()).sum().item()
            correct_beard += (pred_beard == beard.long()).sum().item()
            correct_must += (pred_must == mustache.long()).sum().item()
            correct_color += (pred_color == color).sum().item()
            correct_hair += (pred_hair == hair).sum().item()

    metrics = {
        "val_loss": total_loss / n_batches,
        "acc_glasses": correct_glasses / total_samples,
        "acc_beard": correct_beard / total_samples,
        "acc_mustache": correct_must / total_samples,
        "acc_color": correct_color / total_samples,
        "acc_hair": correct_hair / total_samples,
    }
    return metrics


def train_kfold(dataset, n_splits, epochs, batch_size, lr, device):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    best_global_acc = 0.0
    best_model_uri = None
    best_state_dict = None

    with mlflow.start_run(run_name="kfold_training") as parent_run:
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lr", lr)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"\n===== FOLD {fold} =====")

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            model = CNNMultiTask().to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            with mlflow.start_run(run_name=f"fold_{fold}", nested=True):
                mlflow.log_param("fold", fold)

                for epoch in range(epochs):
                    train_loss = train_one_epoch(model, train_loader, optimizer, device)
                    val_metrics = evaluate(model, val_loader, device)

                    print(
                        f"Fold {fold} | Epoch {epoch} | "
                        f"train_loss={train_loss:.4f} | "
                        f"val_loss={val_metrics['val_loss']:.4f}"
                    )

                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    for k, v in val_metrics.items():
                        mlflow.log_metric(k, v, step=epoch)

                mean_acc = (
                    val_metrics["acc_glasses"]
                    + val_metrics["acc_beard"]
                    + val_metrics["acc_mustache"]
                    + val_metrics["acc_color"]
                    + val_metrics["acc_hair"]
                ) / 5.0

                mlflow.log_metric("mean_acc", mean_acc)

                # log modèle pour ce fold
                mlflow.pytorch.log_model(model, name="model")

                if mean_acc > best_global_acc:
                    best_global_acc = mean_acc
                    current_run_id = mlflow.active_run().info.run_id
                    best_model_uri = f"runs:/{current_run_id}/model"
                    best_state_dict = model.state_dict()

        mlflow.log_metric("best_mean_acc", best_global_acc)
        if best_model_uri is not None:
            mlflow.set_tag("best_model_uri", best_model_uri)

    print("\n[TRAIN] Entraînement k-fold terminé.")
    print("[TRAIN] Meilleure mean_acc =", best_global_acc)
    print("[TRAIN] Meilleur modèle MLflow URI :", best_model_uri)

    return best_state_dict, best_model_uri


# ---------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------

def main():
    # pour la reproductibilité
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1) Prétraitement des images
    print("=== Étape 1 : segmentation + resize des images ===")
    segment_and_resize_images(RAW_IMAGES_DIR, RESIZED_IMAGES_DIR, TARGET_SIZE)

    # 2) Construction du CSV de labels
    print("\n=== Étape 2 : construction du CSV de labels ===")
    build_labels_table(
        METADATA_DIR,
        BEARD_MAP_CSV,
        MUSTACHE_MAP_CSV,
        GLASSES_MAP_CSV,
        HAIR_COLOR_MAP_CSV,
        HAIR_MAP_CSV,
        LABELS_CSV,
    )

    # 3) Dataset
    print("\n=== Étape 3 : création du Dataset ===")
    dataset = FaceDataset(LABELS_CSV, RESIZED_IMAGES_DIR, transform=None)
    print(f"Dataset taille : {len(dataset)} images")

    # 4) Entraînement k-fold + MLflow
    print("\n=== Étape 4 : entraînement k-fold avec MLflow ===")
    best_state_dict, best_model_uri = train_kfold(
        dataset,
        n_splits=N_SPLITS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        device=DEVICE,
    )

    # 5) Sauvegarde locale du meilleur modèle (.pth)
    if best_state_dict is not None:
        model_path = "cnn_multitask_model.pth"
        torch.save(best_state_dict, model_path)
        print(f"\n[SAUVEGARDE] Meilleur modèle sauvegardé dans {model_path}")
    else:
        print("\n[SAUVEGARDE] Aucun state_dict meilleur trouvé (quelque chose a cloché).")


if __name__ == "__main__":
    main()
