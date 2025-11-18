from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import BinaryType
from PIL import Image
import numpy as np
import io
import os
import time

# Dossiers (dans le conteneur)
input_folder = "data/lot1_images"
output_folder = "data/lot1_resized"
target_size = (64, 64)

os.makedirs(output_folder, exist_ok=True)

def segment_and_resize(image_binary):
    # image_binary = contenu binaire de l'image lu par Spark
    img = Image.open(io.BytesIO(image_binary)).convert("RGB")

    gray = img.convert("L")
    arr = np.array(gray)

    # Masque des pixels non-blancs (fond blanc, objets non blancs)
    mask = arr < 250

    if mask.any():
        ys, xs = np.where(mask)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        cropped = img.crop((x_min, y_min, x_max + 1, y_max + 1))
    else:
        cropped = img  # image entièrement blanche → pas de crop

    resized = cropped.resize(target_size)

    buf = io.BytesIO()
    resized.save(buf, format="PNG")
    return buf.getvalue()

segment_udf = udf(segment_and_resize, BinaryType())

if __name__ == "__main__":
    print("Le script a démarré")
    time_start = time.time()

    spark = (
        SparkSession.builder
        .appName("ImageSegmentation")
        .master("local[*]")
        .config("spark.driver.memory", "8g")     # mémoire pour le driver
        .config("spark.executor.memory", "8g")   # mémoire pour les tâches
        .getOrCreate()
    )

    # Lecture des images
    df = spark.read.format("image").load(input_folder)

    # Optionnel : répartir sur plus de partitions pour éviter les gros paquets
    df = df.repartition(16)

    # Application du traitement
    processed_df = df.withColumn("processed", segment_udf(col("image.data")))

    # Sauvegarde progressive (sans collect)
    for row in processed_df.select("image.origin", "processed").toLocalIterator():
        filename = os.path.basename(row["origin"])
        out_path = os.path.join(output_folder, filename)
        with open(out_path, "wb") as f:
            f.write(row["processed"])
        # print("Saved:", out_path)  # décommente si tu veux voir passer les fichiers

    spark.stop()
    time_end = time.time()
    print("Traitement terminé.")
    print("Le temps d'exécution est de :", time_end - time_start, "secondes")
