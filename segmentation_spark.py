from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import BinaryType
from PIL import Image
import numpy as np
import io
import os
import time
print("Le script a démarré")
time_start=time.time()
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
        cropped = img  # image vide → pas de crop

    resized = cropped.resize(target_size)

    buf = io.BytesIO()
    resized.save(buf, format="PNG")
    return buf.getvalue()

segment_udf = udf(segment_and_resize, BinaryType())

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("ImageSegmentation") \
        .master("local[*]") \
        .getOrCreate()

    # Lecture des images : Spark attend un dossier
    df = spark.read.format("image").load(input_folder)

    # Application du traitement
    processed_df = df.withColumn("processed", segment_udf(col("image.data")))

    # Récupération côté Python pour sauvegarde
    rows = processed_df.select("image.origin", "processed").collect()

    for row in rows:
        # origin = chemin complet dans le conteneur -> on ne garde que le nom de fichier
        filename = os.path.basename(row["origin"])
        out_path = os.path.join(output_folder, filename)
        print("-> Saving:", out_path, "size:", len(row["processed"]) if row["processed"] else "None")
        with open(out_path, "wb") as f:
            f.write(row["processed"])

    spark.stop()
    print("Traitement terminé.")
time_end=time.time()
print("Le temps d'exécution est de :",time_end-time_start)
