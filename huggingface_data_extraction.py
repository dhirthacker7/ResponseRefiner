# Import necessary libraries
import os
from io import StringIO
import pandas as pd
from datasets import load_dataset
from google.cloud import storage

#  constants for Hugging Face and GCP
HUGGINGFACE_DATASET = "gaia-benchmark/GAIA"
HUGGINGFACE_CONFIG = "2023_all" 
GCP_BUCKET_NAME = "gaia-benchmark-project1-bucket"
GCP_BUCKET_FOLDER = "huggingface_datasets/"  # Folder path within the bucket
CSV_FILENAME = "gaia_benchmark_1.csv"  

def download_huggingface_dataset(dataset_name, config):
    """
    Downloads the dataset from Hugging Face with the specified configuration and returns it as a pandas DataFrame.
    :param dataset_name: str: Name of the Hugging Face dataset.
    :param config: str: Configuration name for the dataset.
    :return: pandas.DataFrame: The downloaded dataset as a DataFrame.
    """
    try:
        print(f"Downloading dataset '{dataset_name}' with config '{config}' from Hugging Face...")
        dataset = load_dataset(dataset_name, config, split="validation")
        print(f"Successfully downloaded the dataset with {len(dataset)} entries.")
        return dataset.to_pandas()  # Convert the dataset to a pandas DataFrame
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise

def upload_csv_to_gcs(dataframe, bucket_name, bucket_folder, csv_filename):
    """
    Upload the CSV file directly to a Google Cloud Storage bucket without saving it locally.
    :param dataframe: pd.DataFrame: DataFrame to be uploaded as CSV.
    :param bucket_name: str: GCP bucket name.
    :param bucket_folder: str: Folder path within the bucket.
    :param csv_filename: str: Desired CSV file name in the GCP bucket.
    """
    try:
        # Initializing the Google Cloud Storage client
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"{bucket_folder}{csv_filename}")

        if not bucket.exists():
            print(f"Error: The bucket '{bucket_name}' does not exist.")
            return

        # Convert DataFrame to CSV in memory
        print(f"Converting DataFrame to CSV format in memory...")
        csv_buffer = StringIO()
        dataframe.to_csv(csv_buffer, index=False)  # Convert to CSV format without saving locally

        print(f"Uploading CSV data directly to GCS bucket '{bucket_name}' at '{bucket_folder}{csv_filename}'...")
        blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
        print(f"Upload completed: CSV -> gs://{bucket_name}/{bucket_folder}{csv_filename}")
    except Exception as e:
        print(f"Error uploading CSV to GCS: {e}")
        raise

def main():
    try:
        # Step 1: Download the dataset from Hugging Face with the specified configuration
        df = download_huggingface_dataset(HUGGINGFACE_DATASET, HUGGINGFACE_CONFIG)

        # Step 2: Directly upload the DataFrame as a CSV to the GCP bucket
        upload_csv_to_gcs(df, GCP_BUCKET_NAME, GCP_BUCKET_FOLDER, CSV_FILENAME)

        print("Process completed successfully.")
    except Exception as e:
        print(f"Process failed: {e}")

# Run the script
if __name__ == "__main__":
    main()
