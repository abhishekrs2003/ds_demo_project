from src.helper import load_config
import pandas as pd

config = load_config()
raw_data_path = config["raw_data_path"]
data = pd.read_csv(raw_data_path)
print(data.head(2))