import argparse
import random
from pathlib import Path
import sys

#!/usr/bin/env python3
# GitHub Copilot
# pick_up_RandomForAnnote.py
# Choisit N images aléatoires dans lot1 et lot2 et écrit les noms (sans .jpg) une par ligne.

def collect_images(dirs, exts):
    files = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            print(f"Warning: dossier introuvable: {d}", file=sys.stderr)
            continue
        for f in p.iterdir():
            if f.is_file() and f.suffix.lower() in exts:
                files.append(f)
    return files

def main():
    # paramètres en dur (plus d'arguments en ligne de commande)
    args = argparse.Namespace(
        dirs=["data/lot4_images", "data/lot5_images"],
        n=5000,
        out="selected_names.txt",
        exts=[".png"],
        seed=42,
    )

    if args.seed is not None:
        random.seed(args.seed)

    imgs = collect_images(args.dirs, set(e.lower() for e in args.exts))
    total = len(imgs)
    # supprimer les doublons basés sur le nom (sans extension), garder la première occurrence
    seen = set()
    unique_imgs = []
    duplicates = 0
    for p in imgs:
        key = p.stem
        if key not in seen:
            seen.add(key)
            unique_imgs.append(p)
        else:
            duplicates += 1
    if duplicates:
        print(f"{duplicates} doublon(s) supprimé(s) (même nom sans extension).", file=sys.stderr)
    imgs = unique_imgs
    total = len(imgs)

    if total == 0:
        print("Aucune image trouvée. Vérifiez les dossiers et extensions.", file=sys.stderr)
        sys.exit(1)

    k = min(args.n, total)
    if total < args.n:
        print(f"Seulement {total} images trouvées, on sélectionne toutes.", file=sys.stderr)

    selected = random.sample(imgs, k) if k <= total else imgs
    # Écrire les noms sans extension, une par ligne
    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for p in selected:
            f.write(p.stem + "\n")

    print(f"Écrit {k} noms dans {out_path} (à partir de {total} images trouvées).")

if __name__ == "__main__":
    main()