from datasets import load_dataset
import soundfile
import os

# Define path where you want to save the dataset
local_dataset_path = "path/to/minds14_data"
os.makedirs(local_dataset_path, exist_ok=True)

# Load and save the MINDS-14 English dataset with trust_remote_code=True
minds_14 = load_dataset("PolyAI/minds14", 'en-US', trust_remote_code=True)
minds_14.save_to_disk(local_dataset_path)

print(f"Dataset saved to {local_dataset_path}")