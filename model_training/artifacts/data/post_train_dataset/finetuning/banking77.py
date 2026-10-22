from datasets import load_dataset
import pandas as pd



def load_banking_dataset():
    DATASET_NAME = "mteb/banking77"
    print(f"📥 Loading {DATASET_NAME} dataset from Hugging Face...")
    # The dataset natively comes with 'train' and 'test' splits
    dataset = load_dataset(DATASET_NAME)
    print("\n📊 Dataset Structure:")
    print(dataset)
    
    train_df = pd.DataFrame(dataset['train'], columns=dataset['train'].features)
    val_df = pd.DataFrame(dataset['test'], columns=dataset['test'].features)
    
    return train_df, val_df
