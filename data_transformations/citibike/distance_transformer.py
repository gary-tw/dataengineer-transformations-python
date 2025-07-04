from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F

METERS_PER_FOOT = 0.3048
FEET_PER_MILE = 5280
EARTH_RADIUS_IN_METERS = 6371e3
METERS_PER_MILE = METERS_PER_FOOT * FEET_PER_MILE


def compute_distance(_spark: SparkSession, dataframe: DataFrame) -> DataFrame:
    intermediate_df = (
        dataframe
            .withColumn("phi1", F.col("start_station_latitude") * F.pi() / 180)
            .withColumn("phi2", F.col("end_station_latitude") * F.pi() / 180)
            .withColumn("delta_phi", (F.col("end_station_latitude") - F.col("start_station_latitude")) * F.pi() / 180)
            .withColumn("delta_lambda", (F.col("end_station_longitude") - F.col("start_station_longitude")) * F.pi() / 180)
            .withColumn("a",
                        F.pow(F.sin(F.col("delta_phi") / 2), 2) + F.cos(F.col("phi1")) * F.cos(F.col("phi2")) * F.pow(F.sin(F.col("delta_lambda") / 2), 2)
            )
            .withColumn("c", 2 * F.atan2(F.sqrt(F.col("a")), F.sqrt(1 - F.col("a"))))
            .withColumn("distance_in_m", EARTH_RADIUS_IN_METERS * F.col("c"))
            .withColumn("distance", F.col("distance_in_m") / METERS_PER_MILE)
            .withColumn("distance", F.round(F.col("distance"), 2))
    )
    output_df = intermediate_df.drop("phi1", "phi2", "delta_phi", "delta_lambda", "a", "c", "distance_in_km")

    return output_df

def run(spark: SparkSession, input_dataset_path: str, transformed_dataset_path: str) -> None:
    input_dataset = spark.read.parquet(input_dataset_path)
    input_dataset.show()

    dataset_with_distances = compute_distance(spark, input_dataset)
    dataset_with_distances.show()

    dataset_with_distances.write.parquet(transformed_dataset_path, mode='append')
