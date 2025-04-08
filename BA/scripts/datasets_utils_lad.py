import os
import torch
import pandas as pd
import numpy as np
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm

###########################################################
# functions which can be applyed for all datasets classes #
###########################################################

def pad_sequence(batch):
            # Make all tensor in a batch the same length by padding with zeros
            batch = [item.t() for item in batch]
            batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
            return batch

def load_list(root, person_list):
            output = []
            labels = []
            for person in person_list:
                filepath = os.path.join(root, person)
                for files in os.listdir(filepath):
                    #print(files)
                    #split by _ and take the first element
                    labels.append(int(files.split('_')[0]))
                    output.append(os.path.normpath(os.path.join(filepath, files)))
            return output, labels

def to_mono_if_stereo(waveform):
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform
    else:
        return waveform

def resample_if_not_sample_rate(waveform, sr, sample_rate):
    if sr == sample_rate:
        return waveform
    else:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
        return waveform


###############################
###### UrbanSoundDataset ######
###############################

class UrbanSoundDataset(Dataset):
    def __init__(self, metadata_file, audio_dir, subset: str = None, download: bool = False, 
                 device='cpu', sample_rate=16000, num_samples=160000, transform=None, 
                 path='data\\UrbanSound8K', custom_metadata=None):
        # device is the device where the data should be stored
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.transform = transform
        self.device = device

        # Load full metadata
        metadata = pd.read_csv(metadata_file)

        # map class labels to integers
        self.named_labels = sorted(metadata['class'].unique())
        self.class_mapping = {label: idx for idx, label in enumerate(self.named_labels)}

        # If custom_metadata is provided, use it instead of filtering by subset
        if custom_metadata is not None:
            self.metadata = custom_metadata
        # Otherwise, split data according to subset
        elif subset == "validation": # fold 9
            self.metadata = metadata[metadata['fold'] == 9]
        elif subset == "testing": # fold 10
            self.metadata = metadata[metadata['fold'] == 10]
        elif subset == "training":
            self.metadata = metadata[metadata['fold'].isin(range(1, 9))]
        elif subset == "custom":
            # Empty metadata for custom_metadata case, will be populated later
            self.metadata = pd.DataFrame()
        else:
            raise ValueError("subset should be 'training', 'validation', 'testing', or 'custom'")
        
        # generate file paths based on metadata
        self.walker = [os.path.normpath(os.path.join(
            self.audio_dir, 
            f"fold{row['fold']}", 
            row['slice_file_name']
        )) for _, row in self.metadata.iterrows()]

        # create cache path if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)
        
        # For custom subset, generate a unique identifier based on fold information
        cache_id = ""
        if custom_metadata is not None:
            folds = sorted(custom_metadata['fold'].unique())
            cache_id = f"_folds_{'_'.join(map(str, folds))}"
        
        cache_path_audio = os.path.join(path, f"{subset}{cache_id}_audios.pt")
        cache_path_labels = os.path.join(path, f"{subset}{cache_id}_labels.pt")
        
        # load or create cached data
        if os.path.exists(cache_path_audio) and os.path.exists(cache_path_labels):
            self.audios = torch.load(cache_path_audio, weights_only=True)
            self.labels = torch.load(cache_path_labels, weights_only=True)
        else:
            self.audios = []
            self.labels = []
            
            for item in tqdm(self.walker, desc=f"Loading {subset}{cache_id} set"):
                waveform, sr = torchaudio.load(item)
                waveform = to_mono_if_stereo(waveform)
                waveform = resample_if_not_sample_rate(waveform, sr, sample_rate)
                
                self.audios.append(waveform)
                
                # get label from metadata
                file_name = os.path.basename(item)
                label_row = self.metadata[self.metadata['slice_file_name'] == file_name].iloc[0]
                label = self.class_mapping[label_row['class']]
                self.labels.append(label)
        
            self.audios = pad_sequence(self.audios)
            self.labels = torch.tensor(self.labels)
            
            # normalize using training set statistics
            if subset != "training" and "training" not in cache_id:
                mean_path = os.path.join(path, "mean.pt")
                std_path = os.path.join(path, "std.pt")
                
                if not os.path.exists(mean_path) or not os.path.exists(std_path):
                    # this will trigger training set calculation if not already done
                    UrbanSoundDataset(metadata_file, audio_dir, "training", device=device, 
                                    sample_rate=sample_rate, num_samples=num_samples, path=path)
                self.mean = torch.load(mean_path)
                self.std = torch.load(std_path)
            else:
                self.mean = self.audios.mean()
                self.std = self.audios.std()
                torch.save(self.mean, os.path.join(path, "mean.pt"))
                torch.save(self.std, os.path.join(path, "std.pt"))
                
            print(f"Dataset mean={self.mean} std={self.std}")
            self.audios = (self.audios - self.mean) / (self.std + 1e-5)
                
            torch.save(self.audios, cache_path_audio)
            torch.save(self.labels, cache_path_labels)
            
        self.audios = self.audios.to(device)
        self.labels = self.labels.to(device)

    def __len__(self) -> int:
        return len(self.walker)
    
    def get(self, idx):
        return self.audios[idx,...], self.labels[idx,...]
    
    def __getitem__(self, idx):
        if self.device != 'cpu':
            return idx
        
        audio = self.audios[idx].clone()  # clone to avoid modifying the original
        label = self.labels[idx]
        
        # for training, we can apply additional random transformations here
        # like random cropping for variable length inputs
        
        if self.transform:
            audio = self.transform(audio)
            
        return audio, label