import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from video_llama.models.ImageBind.data import load_and_transform_audio_data
from video_llama.processors.video_processor import load_video, ToTHWC, ToUint8
from video_llama.processors import transforms_video

IMG_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMG_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_SIZE = 480


class VideoAudioDataset(Dataset):

    def __init__(self, dataset_csv, num_candidates=10, device='cuda:0'):

        self.device = device
        self.num_candidates = num_candidates

        self.clip_df = pd.read_csv(dataset_csv)
        self.clip_df = self.clip_df.sample(frac=1.0, replace=False, random_state=0)

        self.video_pipeline = transforms.Compose([
            ToTHWC(),
            ToUint8(),
            transforms_video.ToTensorVideo(),
            transforms_video.NormalizeVideo(IMG_MEAN, IMG_STD)
        ])

        # cache all items
        self.video_cache = []
        self.audio_cache = []

        for idx, row in self.clip_df.iterrows():

            # load video tensor
            clip_path = row['clip_path']
            # leaving n_frms as MAX_INT in the below
            video = load_video(
                clip_path,
                height=-1,
                width=-1,
                sampling='uniform',
                return_msg=False  # True to display number of frames and sampling interval
            )
            video = self.video_pipeline(video)
            self.video_cache.append(video)

            # load audio tensor
            audio_path = row['audio_path']
            audio = load_and_transform_audio_data(
                audio_path, 
                device='cpu',
                num_mel_bins=128, 
                target_length=204,
                sample_rate=16000,
                clip_duration=2,
                clips_per_video=5,  # to use all audio in each 10 sec clip
                mean=-4.268,
                std=9.138,
            )
            self.audio_cache.append(audio)

        print('Dataset cached')

    # return video tensor, plus set of candidate audios (random)
        # where first candidate audio is ground truth
    # assuming all video/audio is already cached!
    def __getitem__(self, idx):

        video = self.video_cache[idx].to(self.device)

        audios = [self.audio_cache[idx].to(self.device)]
        candidate_indices = np.random.randint(low=0, high=len(self.clip_df)-1, size=(self.num_candidates-1))
        for candidate_idx in candidate_indices:
            if candidate_idx >= idx:
                candidate_idx += 1
            audios.append(self.audio_cache[candidate_idx].to(self.device))
        audios = torch.stack(audios, dim=0)

        return video, audios

    def __len__(self):
        return len(self.clip_df)

    # NOTE: collation ??
