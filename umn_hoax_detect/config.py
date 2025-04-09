import os

# Path to your Mafindo CSV dataset
DATASET_PATH = os.getenv("DATASET_PATH", "hoax_1k.csv")

# Columns to extract
COLUMNS = ["title", "content", "fact", "conclusion"]
