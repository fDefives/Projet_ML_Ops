#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
inference.py
Utilise le modèle entraîné (cnn_multitask_model.pth) pour prédire
sur des images non labellisées.

Pipeline :
  1. Segmentation + resize des images du dossier de test
  2. Dataset sans labels
  3. Chargement du modèle .pth
  4. Prédictions -> predictions.csv
  5. Construction du vecteur final -> final_predictions.csv
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

# 🔁 Import depuis ton script d'entraînement
from train_ml_flow import CNNMultiTask, segment_and_resize_images, TARGET_SIZE

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Dossier avec les images brutes à prédire
RAW_TEST_DIR = "data/lot7_images"
# Dossier où seront stockées les images segmentées + redimensionnées
RESIZED_TEST_DIR = "data/lot7_resized"

# Modèle entraîné
MODEL_PATH = "cnn_multitask_best_final.pth"

# Batch size pour l'inférence
BATCH_SIZE = 64

# Fichiers de sortie
PREDICTIONS_CSV = "predictions.csv"
FINAL_PREDICTIONS_CSV = "G6_L7.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# DATASET SANS LABELS
# ---------------------------------------------------------------------

class FaceDatasetNoLabels(Dataset):
    """
    Dataset qui lit uniquement les images d'un dossier (jpg/png)
    et renvoie :
      - image : tensor (C,H,W)
      - filename : nom du fichier image
    """

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if len(self.images) == 0:
            raise RuntimeError(f"Aucune image trouvée dans {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        else:
            # même comportement que dans script.py si transform=None
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0

        return {"image": image, "filename": img_name}


# ---------------------------------------------------------------------
# FONCTION PRINCIPALE D'INFERENCE
# ---------------------------------------------------------------------

def run_inference():
    # 1) Prétraitement des images de test
    print("=== Étape 1 : segmentation + resize des images de test ===")
    os.makedirs(RESIZED_TEST_DIR, exist_ok=True)
    segment_and_resize_images(RAW_TEST_DIR, RESIZED_TEST_DIR, TARGET_SIZE)

    # 2) Chargement du modèle
    print("\n=== Étape 2 : chargement du modèle .pth ===")
    model = CNNMultiTask().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Modèle chargé depuis {MODEL_PATH}")

    # 3) Dataset et DataLoader
    print("\n=== Étape 3 : création du DataLoader de test ===")
    test_dataset = FaceDatasetNoLabels(RESIZED_TEST_DIR, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"{len(test_dataset)} images à prédire")

    # 4) Fichier de sortie brut (one-hot)
    print("\n=== Étape 4 : génération de predictions.csv ===")
    with open(PREDICTIONS_CSV, "w") as f:
        # ⚠️ ordre des colonnes cohérent avec nos one-hot :
        # couleurs : blond, light_brown, redhead, dark_brown, gray_blue
        # cheveux : bald, short, long
        f.write(
            "filename,glasses,beard,mustache,"
            "blond,light_brown,redhead,dark_brown,gray_blue,"
            "bald,short,long\n"
        )

    with torch.no_grad():
        for batch in test_loader:
            # print(f"Traitement du batch {test_loader._index}/{len(test_loader)}", end="\r")
            images = batch["image"].to(DEVICE)
            filenames = batch["filename"]

            glasses_logits, beard_logits, must_logits, color_logits, hair_logits = model(images)

            # binaires (on a un logit par échantillon)
            pred_glasses = (torch.sigmoid(glasses_logits).view(-1) > 0.5).int().cpu().numpy()
            pred_beard = (torch.sigmoid(beard_logits).view(-1) > 0.5).int().cpu().numpy()
            pred_mustache = (torch.sigmoid(must_logits).view(-1) > 0.5).int().cpu().numpy()

            # multi-classes
            pred_color = torch.argmax(color_logits, dim=1).cpu().numpy()  # 0..4
            pred_hair = torch.argmax(hair_logits, dim=1).cpu().numpy()    # 0..2

            # one-hot cohérents avec l'ordre des classes en entraînement
            # couleur: index -> [blond, light_brown, redhead, dark_brown, gray_blue]
            pred_color_onehot = np.eye(5, dtype=int)[pred_color]
            # cheveux: index -> [bald, short, long]
            pred_hair_onehot = np.eye(3, dtype=int)[pred_hair]

            # écriture dans le CSV
            with open(PREDICTIONS_CSV, "a") as f:
                for i, filename in enumerate(filenames):
                    row = [filename]
                    row.append(pred_glasses[i])
                    row.append(pred_beard[i])
                    row.append(pred_mustache[i])
                    row.extend(pred_color_onehot[i])  # 5 valeurs
                    row.extend(pred_hair_onehot[i])   # 3 valeurs
                    f.write(",".join(map(str, row)) + "\n")

    print(f"Predictions brutes sauvegardées dans {PREDICTIONS_CSV}")

    # 5) Construction du vecteur final
    print("\n=== Étape 5 : construction de final_predictions.csv ===")

    predictions_df = pd.read_csv(PREDICTIONS_CSV)

    # Correspondance des valeurs pour taille_cheveux et couleur_cheveux
    taille_cheveux_mapping = {"bald": 0, "short": 1, "long": 2}
    couleur_cheveux_mapping = {
        "blond": 0,
        "light_brown": 1,
        "redhead": 2,
        "dark_brown": 3,
        "gray_blue": 4,
    }

    # On récupère la colonne avec la valeur 1 et on mappe vers l'index
    predictions_df["taille_cheveux"] = (
        predictions_df[["bald", "short", "long"]]
        .idxmax(axis=1)
        .map(taille_cheveux_mapping)
    )
    predictions_df["couleur_cheveux"] = (
        predictions_df[["blond", "light_brown", "redhead", "dark_brown", "gray_blue"]]
        .idxmax(axis=1)
        .map(couleur_cheveux_mapping)
    )

    # Vecteur final (mêmes noms que tu avais)
    final_df = predictions_df.rename(columns={
        "filename": "image_name",
        "beard": "barbe",
        "mustache": "moustache",
        "glasses": "lunettes",
    })[["image_name", "barbe", "moustache", "lunettes", "taille_cheveux", "couleur_cheveux"]]
    # Suppression de l'extension .png dans la colonne image_name
    final_df["image_name"] = final_df["image_name"].str.replace(".png", "", regex=False)

    print(final_df.head())
    final_df.to_csv(FINAL_PREDICTIONS_CSV, index=False)
    print(f"\nVecteur final sauvegardé dans {FINAL_PREDICTIONS_CSV}")


# ---------------------------------------------------------------------

if __name__ == "__main__":
    run_inference()
