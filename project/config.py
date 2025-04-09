import os

# Path to your Mafindo CSV dataset
DATASET_PATH = os.getenv("DATASET_PATH", "data/mafindo_hoax_data.csv")

# Columns to extract
COLUMNS = ["title", "content", "fact", "conclusion"]
