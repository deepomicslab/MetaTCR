import torch
from .tcr2vec.model import TCR2vec
from .tcr2vec.dataset import TCRLabeledDset
from .tcr2vec.utils import get_emb
from torch.utils.data import DataLoader

import numpy as np
import os

def check_model_exist(path_to_TCR2vec = '../../pretrained_models/TCR2vec_120'):

    model_file_path = os.path.join(path_to_TCR2vec, 'pytorch_model.bin')
    args_file_path = os.path.join(path_to_TCR2vec, 'args.json')
    config_file_path = os.path.join(path_to_TCR2vec, 'config.json')

    # Check if model and json files exist
    if not os.path.exists(model_file_path) or not os.path.exists(args_file_path) or not os.path.exists(config_file_path):
        import zipfile
        
        pretrained_path = os.path.dirname(path_to_TCR2vec)
        download_url = 'https://drive.google.com/uc?export=download&id=1Nj0VHpJFTUDx4X7IPQ0OGXKlGVCrwRZl'
        zip_path = os.path.join(pretrained_path, 'tcr2vec_120.zip')

        print(f"Pretrained model not found at {path_to_TCR2vec}")
        
        try:
            import gdown
            os.makedirs(pretrained_path, exist_ok=True)
            print(f"Attempting to download model using gdown from {download_url} to {zip_path}")
            
            try:
                output = gdown.download(download_url, zip_path, quiet=False)
                # Check if file was actually downloaded (gdown might not raise exception but return None/fail silently)
                if not output or not os.path.exists(zip_path):
                     raise Exception("Download failed or file not created.")

                print(f"Extracting {zip_path} to {pretrained_path}")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(pretrained_path)
                os.remove(zip_path)
                print(f"TCR encoder model TCR2vec extracted successfully.")
                
            except Exception as e:
                print(f"\nAutomatic download failed: {e}")
                print(f"Please manually download the model from: {download_url}")
                print(f"And place/extract it to: {pretrained_path}")
                if os.path.exists(zip_path): # clean up partial download
                    os.remove(zip_path)
                # We raise an error here because the model is required for the next steps
                raise RuntimeError("Model download failed. Manual download required.")

        except ImportError:
            print(f"\nThe 'gdown' package is not installed for automatic downloading.")
            print(f"Please manually download the model from: {download_url}")
            print(f"And place/extract it to: {pretrained_path}")
            print(f"Alternatively, install gdown (pip install gdown) to try automatic download.")
            raise RuntimeError("Model not found and gdown not installed. Manual download required.")
    else:
        print(f"loading TCR2vec encoder...")


## load the trained TCR2vec model
def load_tcr2vec(path_to_TCR2vec = '../../pretrained_models/TCR2vec_120', device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    check_model_exist(path_to_TCR2vec)
    emb_model = TCR2vec(path_to_TCR2vec)
    emb_model = emb_model.to(device)
    return emb_model

def seqlist2ebd(seq_list, emb_model, emb_size = 120, keep_pbar = True):  ## input: a list of TCR seqs ['CAAAGGIYEQYF', 'CAAAPGINEQFF' ... ], output: the mtx of 96-dim embedding

    if len(seq_list) == 0 :
        return np.zeros((1, emb_size),dtype='float32')

    dset = TCRLabeledDset(seq_list, only_tcr=True) #input a list of TCRs
    loader = DataLoader(dset, batch_size=2048, collate_fn=dset.collate_fn, shuffle=False)
    emb = get_emb(emb_model, loader, detach=True, keep_pbar = keep_pbar) #B x emb_size

    return emb


if __name__ == "__main__":
    ## load the trained TCR2vec model
    path_to_TCR2vec = '../../pretrained_models/TCR2vec_120'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    emb_model = load_tcr2vec(path_to_TCR2vec, device)

    # convert list of seqs to numpy array
    seq_list = ['NAGVTQTPKFQVLKTGQSMTLQCAQDMNHNSMYWYRQDPYSASEGTTDKGEVPNGYNVSRLNKREFSLRLESAAPSQTSVYFCASSEALGTGNTIYFGEGSWLTVV',
                'NAGVTQTPKFQVLKTGQSMTLQCAQDMNHNSMYWYRQDPGMGLLLIYYSASEGTTDKGEVPNGYNVSRLNKREFSLRLESAAPSQTSVYFCASSEALGTGNTIYFGEGSWLTVV',
                'NAGVTQTPKFQVLKTGQSMTLQCAQDMNHNSMYWYRQDPGMGLRLIYYSRLNKREFSLRLESAAPSQTSVYFCASSEALGTGNTIYFGEGSWLTVV']
    embmtx = seqlist2ebd(seq_list, emb_model)
    print("example seq list = ", seq_list)
    print("embedding mtx shape = ", embmtx.shape)
    print("embedding mtx = ", embmtx)
