from src.data_preprocessing import preprocess_data
from src.model_training import train_model
import pandas as pd

# Load raw dataset
raw_df = pd.read_csv("data/flight_delay.csv")

# Preprocess
df = preprocess_data("data/flight_delay.csv")

# Save processed dataset (VERY IMPORTANT)
df.to_csv("data/processed_data.csv", index=False)

# Train model
train_model(df)

print("Model trained successfully!")
