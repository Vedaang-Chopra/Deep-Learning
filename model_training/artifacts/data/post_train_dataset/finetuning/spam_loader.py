import requests
import zipfile
import os
from pathlib import Path
import pandas as pd

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_file = "sms_spam_collection.zip"
extracted_dir = "sms_spam_collection"
bkp_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
        

def download_and_unzip_spam_data(data_dir_path:Path, url=url):
    
    zip_file_path = data_dir_path / "downloads" 
    os.makedirs(zip_file_path, exist_ok=True)
    
    extracted_path = data_dir_path /  extracted_dir 
    os.makedirs(extracted_path, exist_ok=True)
        
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"
    
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return data_file_path

    # Downloading the file
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(zip_file_path / zip_file, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)

    # Unzipping the file
    with zipfile.ZipFile(zip_file_path / zip_file, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")
    return data_file_path

def load_dataset(data_dir_path: Path):
    try:
        data_file_path = download_and_unzip_spam_data(data_dir_path, url=url)
    except (requests.exceptions.RequestException, TimeoutError) as e:
        print(f"Primary URL failed: {e}. Trying backup URL...")
        data_file_path = download_and_unzip_spam_data(data_dir_path, url= bkp_url, )

    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])    
    return df


def create_balanced_dataset(df):
    
    # Count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]
    
    # Randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    
    # Combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df


# Let's now define a function that randomly divides the dataset into training, validation, and test subsets
def random_split(df, train_frac, validation_frac):
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

