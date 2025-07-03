import logging

from pyspark.sql import SparkSession
import pyspark.sql.functions as F


def run(spark: SparkSession, input_path: str, output_path: str) -> None:
    logging.info("Reading text file from: %s", input_path)
    input_df = spark.read.text(input_path)

    logging.info("Writing csv to directory: %s", output_path)

    output_df = (
        input_df
        .withColumn("value", F.lower(F.col("value")))
        .withColumn("value", F.regexp_replace(F.col("value"), "[^\w\s+']", " "))
        .withColumn("word", F.explode(F.split(F.col("value"), "\s+")))
        .drop("value")
        .filter(F.length(F.col("word")) > 0)
        .groupby("word")
        .count()
        .orderBy("word")
    )

    output_df.coalesce(1).write.csv(output_path, header=True)
