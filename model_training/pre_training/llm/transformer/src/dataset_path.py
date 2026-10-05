import torch 
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE


def find_project_root(start=None, markers=(".gitignore",)):
    p = Path(start or Path.cwd()).resolve()
    for cur in [p, *p.parents]:
        if any((cur / m).exists() for m in markers):
            return cur
    return p


BASE_CODE_DIR_PATH = find_project_root()
DATASET_DIR = BASE_CODE_DIR_PATH / 'datasets'


# Hugging Face resolves this to its user-level default cache unless overridden
# by a standard Hugging Face environment variable.
HF_CACHE_DIR = Path(HF_HUB_CACHE)



def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask==0

class BilingualDataset(Dataset):
    
    def __init__(self, 
                ds,
                tokenizer_src,
                tokenizer_tgt,
                src_lang,
                tgt_lang,
                seq_len,
                *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ds = ds
        self.seq_len = seq_len
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        # self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        # self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)
        
        self.sos_token_src = torch.tensor(
            [tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token_src = torch.tensor(
            [tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token_src = torch.tensor(
            [tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64
        )

        self.sos_token_tgt = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token_tgt = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token_tgt = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64
        )
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        ## Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) -2 # We will add SOS and EOS
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) -1 # We will add SOS 
        
        
        ## Make sure the no of padding tokens is not negative. If it is the sentence is too long
        if enc_num_padding_tokens <0 or dec_num_padding_tokens <0:
            raise ValueError("Sentence is too long")
        
        ### Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token_src,
                torch.tensor(enc_input_tokens, dtype = torch.int64),
                self.eos_token_src,
                torch.tensor([self.pad_token_src]* enc_num_padding_tokens, dtype = torch.int64),
            ],
            dim=0
        )
        decoder_input = torch.cat(
            [
                self.sos_token_tgt,
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token_tgt]* dec_num_padding_tokens, dtype = torch.int64),
            ],
            dim=0
        )
        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token_tgt,
                torch.tensor([self.pad_token_tgt] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        return {
            "encoder_input" :encoder_input ,
            "decoder_input" :decoder_input,
            "encoder_mask" :(encoder_input !=self.pad_token_src).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask" :(decoder_input !=self.pad_token_tgt).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
            
        }
                

        
        
        
        
        
        
        
    




def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    
    ### Here we are building the tokenizer for our dataset
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        
        tokenizer = Tokenizer(WordLevel(unk_token = "[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens =["[UNK]", "[SOS]", "[EOS]", "[PAD]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
    return tokenizer




def complete_tokenization(config, ds_raw):
    if ds_raw == None:
        ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    
    ## Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    ### Splitting Data into Training and Testing 
    train_ds_size = int(0.9*len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    
    
    #### Using Pytorch DataLoader
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    
    ## Max Sequence Length
    max_len_src =0
    max_len_tgt =0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
        
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f"Max Length of Source Sentence: {max_len_src}")
    print(f"Max Length of Target Sentence: {max_len_tgt}")
    
    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle = True)
    
    return train_dataloader,val_dataloader, tokenizer_src, tokenizer_tgt
        
    
    
    
