from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import time
import os
import pandas as pd

# initialisation Spark
spark = SparkSession.builder.appName("CSVProcessing").getOrCreate()

time_start = time.time()

input_folder = 'data/lot1_images/'
output_folder = 'data/lot1_resized/'

# Chargement des CSV "références" via pandas (pas besoin Spark ici)
beard = pd.read_csv("beard.csv", sep=";")
mustache = pd.read_csv("mustache.csv", sep=";")
glasses = pd.read_csv("glasses.csv", sep=";")
hair_color = pd.read_csv("hair_color.csv", sep=";")
hair = pd.read_csv("hair.csv", sep=";")

# listes pour lookup
glasses_yes = set(glasses['yes'].dropna())
beard_yes = set(beard['yes'].dropna())
mustache_yes = set(mustache['yes'].dropna())

hc_blond = set(hair_color['blond'].dropna())
hc_light_brown = set(hair_color['light_brown'].dropna())
hc_dark_brown = set(hair_color['dark_brown'].dropna())
hc_redhead = set(hair_color['redhead'].dropna())
hc_gray_blue = set(hair_color['gray_blue'].dropna())

h_long = set(hair['long'].dropna())
h_short = set(hair['short'].dropna())
h_bald = set(hair['bald'].dropna())

# boucle sur les fichiers CSV via Spark
for file in os.listdir(input_folder):
    if file.endswith(".csv"):

        # lecture Spark
        df = spark.read.option("header", "false").option("delimiter", ",").csv(os.path.join(input_folder, file))
        df = df.withColumnRenamed("_c0", "key").withColumnRenamed("_c1", "value")
        
        # nettoyage comme dans votre code
        df = df.withColumn("key", F.trim(F.regexp_replace("key", '"', ''))) \
               .withColumn("value", F.trim(F.regexp_replace("value", '"', "")))
        # récupération en local pour logique identique
        pdf = df.toPandas()

        # ======== LOGIQUE IDENTIQUE À VOTRE CODE =========

        row = pdf[pdf["key"] == "glasses"]
        value = int(row.iloc[0, 1])
        glasses_value = 1 if value in glasses_yes else 0

        row = pdf[pdf["key"] == "facial_hair"]
        value = int(row.iloc[0, 1])
        beard_value = 1 if value in beard_yes else 0
        mustache_value = 1 if value in mustache_yes else 0

        row = pdf[pdf["key"] == "hair_color"]
        value = int(row.iloc[0, 1])
        blond_value = 1 if value in hc_blond else 0
        light_brown_value = 1 if value in hc_light_brown else 0
        dark_brown_value = 1 if value in hc_dark_brown else 0
        redhead_value = 1 if value in hc_redhead else 0
        gray_blue_value = 1 if value in hc_gray_blue else 0

        row = pdf[pdf["key"] == "hair"]
        value = int(row.iloc[0, 1])
        long_value = 1 if value in h_long else 0
        short_value = 1 if value in h_short else 0
        bald_value = 1 if value in h_bald else 0

        # écriture du fichier résultant
        output_file = os.path.join(output_folder, file)
        with open(output_file, 'w') as f:
            f.write(f"glasses;{glasses_value}\n")
            f.write(f"beard;{beard_value}\n")
            f.write(f"mustache;{mustache_value}\n")
            f.write(f"blond;{blond_value}\n")
            f.write(f"light_brown;{light_brown_value}\n")
            f.write(f"dark_brown;{dark_brown_value}\n")
            f.write(f"redhead;{redhead_value}\n")
            f.write(f"gray_blue;{gray_blue_value}\n")
            f.write(f"long;{long_value}\n")
            f.write(f"short;{short_value}\n")
            f.write(f"bald;{bald_value}\n")

time_end = time.time()
print(f"Temps de traitement des .csv : {time_end - time_start} secondes")
