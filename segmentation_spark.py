from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import BinaryType
from PIL import Image, UnidentifiedImageError
import numpy as np
import io
import os
import time

# Dossiers (dans le conteneur) 
input_folder = "data/lot4_images"
output_folder = "data/lot4_resized"
target_size = (64, 64)

os.makedirs(output_folder, exist_ok=True)

def segment_and_resize(image_binary):
    try:
        img = Image.open(io.BytesIO(image_binary)).convert("RGB")
    except UnidentifiedImageError:
        # Si l'image est corrompue ou non lisible
        return None

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
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .getOrCreate()
    )

    # Lecture des images PNG
    df = spark.read.format("binaryFile") \
        .option("recursiveFileLookup", "true") \
        .load(input_folder) \
        .filter(col("length") > 0) \
        .filter(col("path").endswith(".png"))

    df = df.repartition(16)

    processed_df = df.withColumn("processed", segment_udf(col("content")))

    # Sauvegarde progressive
    for row in processed_df.select("path", "processed").toLocalIterator():
        if row["processed"] is not None:
            filename = os.path.basename(row["path"])
            out_path = os.path.join(output_folder, filename)
            with open(out_path, "wb") as f:
                f.write(row["processed"])

    spark.stop()
    time_end = time.time()
    print("Traitement terminé.")
    print("Temps d'exécution :", time_end - time_start, "secondes")
