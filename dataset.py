import pandas as pd
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torch

class AlbumCoverArtDataset(Dataset):
    def __init__(self, album_root_path, album_metadata_file):
        metadata = pd.read_csv(album_metadata_file)
        
        self.album_names = list(metadata['album_name'])
        self.genres = list(metadata['genres'])
        self.singular_genres = list(metadata['singular_genre'])
        self.album_ids = list(metadata['album_id'])
        self.song_ids = list(metadata['song_id'])
        self.artists = list(metadata['artist'])

        album_images = []
        for album_id in self.album_ids:
            img = Image.open(os.path.join(album_root_path, f"{album_id}.jpg")).convert('RGB')
            album_images.append(img)
        
        transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((512, 512)),
                torchvision.transforms.ToTensor(),
                ])
        self.album_images = torch.stack([transforms(img) for img in album_images])
    
    def __len__(self):
        return len(self.album_names)
    
    def __getitem__(self, index):
        if self.genres[index].find(',') != -1:
            s = f"""Create an album cover for the album '{self.album_names[index]}'. The genres are {self.genres[index]}."""
        else:
            s = f"""Create an album cover for the album '{self.album_names[index]}'. The genre is {self.genres[index]}."""
        return s, self.album_images[index], self.album_ids[index]
        