import pandas as pd
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torch
import numpy as np

class AlbumCoverArtDataset(Dataset):
    def __init__(self, album_root_path, album_metadata_file, audio_embedding_folder=None):
        metadata = pd.read_csv(album_metadata_file)
        
        self.album_root_path = album_root_path
        self.album_names = list(metadata['album_name'])
        self.genres = list(metadata['genres'])
        self.singular_genres = list(metadata['singular_genre'])
        self.album_ids = list(metadata['album_id'])
        self.song_ids = list(metadata['song_id'])
        self.artists = list(metadata['artist'])

        self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.ToTensor(),
                ])
        
        self.audio_embedding_folder = audio_embedding_folder

    def __len__(self):
        return len(self.album_names)
    
    def __getitem__(self, index):
        s = f"""Create an album cover for the album '{self.album_names[index]}'. The genre is {self.singular_genres[index]}."""

        img = Image.open(os.path.join(self.album_root_path, f"{self.album_ids[index]}.jpg")).convert('RGB')     
        img = self.transforms(img)

        if self.audio_embedding_folder is None:
            return s, img, self.album_ids[index]
        else:
            audio_embeddings = torch.from_numpy(np.load(os.path.join(self.audio_embedding_folder, f"{self.song_ids[index]}.npy")))
            return s, img, self.album_ids[index], audio_embeddings
        