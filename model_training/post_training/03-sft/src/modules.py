
from functools import partial
from torch.utils.data import DataLoader



def data_prepare(data_dir, file_name, url):
    data = download_and_load_file(data_dir=(data_dir / "instruction_tuning"), file_name=file_name, url = url)
    print("Number of entries:", len(data))
    
    dataset  = hf_data.from_list(data)
    sft_data= SFTDatasetPrepare(model_name=MODEL_NAME, template="model",)
    prepared = sft_data.prepare(dataset)
    
    return sft_data, prepared


def data_split(prepared, train_portion, test_portion)
    
    train_data = prepared.select(range(train_portion))
    test_data = prepared.select(
        range(train_portion, train_portion + test_portion)
    )
    val_data = prepared.select(
        range(train_portion + test_portion, len(prepared))
    )

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))
    
    
    train_dataset = InstructionDataset(
    ds=train_data,
    tokenizer=tokenizer,
    prompt_style=sft_data,
    )
    val_dataset = InstructionDataset(
    ds=val_data,
    tokenizer=tokenizer,
    prompt_style=sft_data,
    )
    test_dataset = InstructionDataset(
        ds=test_data,
        tokenizer=tokenizer,
        prompt_style=sft_data,
    )

    return train_dataset, val_dataset, test_dataset



def prepare_dataloader(train_dataset, 
                    val_dataset,
                    test_dataset, 
                    device, 
                    pad_token_id,
                    num_workers, batch_size):
        
    # Changing the defaults at runtime
    custom_collate_fn1 = partial(custom_collate_fn, device=device, allowed_max_length = 1024, pad_token_id=pad_token_id)
    
    


    train_loader = create_instruction_dataloader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn1,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )
    val_loader = create_instruction_dataloader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn1,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    test_loader = create_instruction_dataloader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn1,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    return train_loader, val_loader, test_loader
    

def return_dataset(data_dir, file_name, url):
    sft_data, prepared = data_prepare(data_dir, file_name, url)
    
    train_dataset, val_dataset, test_dataset= data_split(prepared, train_portion, test_portion)

    train_loader, val_loader, test_loader= prepare_dataloader(train_dataset, 
                    val_dataset,
                    test_dataset, 
                    device, 
                    pad_token_id,
                    num_workers, batch_size)
    return train_loader, val_loader, test_loader
