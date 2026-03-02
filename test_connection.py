from databricks.connect import DatabricksSession
from dotenv import load_dotenv
import os

load_dotenv()

spark = DatabricksSession.builder \
    .host(os.getenv("DATABRICKS_HOST")) \
    .token(os.getenv("DATABRICKS_TOKEN")) \
    .clusterId(os.getenv("DATABRICKS_CLUSTER_ID")) \
    .getOrCreate()

# Simple test
df = spark.range(5)
df.show()

print("✅ Databricks Connect working!")
print(f"Spark version: {spark.version}")