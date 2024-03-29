import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from video_llama.models.ImageBind.data import load_and_transform_audio_data
from video_llama.processors.video_processor import load_video, ToTHWC, ToUint8
from video_llama.processors import transforms_video

IMG_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMG_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_SIZE = 480


class VideoAudioDataset(Datset):

    def __init__(self, dataset_csv, device='cuda:0'):

        self.device = device

        self.clip_df = pd.read_csv(dataset_csv)
        self.clip_df = self.clip_df.sample(frac=1.0, replace=False, random_state=0)

        self.video_pipeline = transforms.Compose([
            ToTHWC(),
            ToUint8(),
            transforms_video.ToTensorVideo(),
            transforms_video.NormalizeVideo(IMG_MEAN, IMG_STD)
        ])

        # cache items keyed by index in df
        # cached on cpu to not take up gpu mem
        self.video_cache = {}
        self.audio_cache = {}

    def __getitem__(self, idx):

        video = None
        audio = None

        if idx in self.video_cache.keys():
            video = self.video_cache[idx]
        if idx in self.audio_cache.keys():
            audio = self.audio_cache[idx]

        if video is None:
            clip_path = self.clip_df.iloc[idx]['clip_path']
            # leaving n_frms as MAX_INT in the below
            video = load_video(
                clip_path,
                height=-1,
                width=-1,
                sampling='uniform',
                return_msg=False  # True to display number of frames and sampling interval
            )
            video = self.video_pipeline(video)
            self.video_cache[idx] = video

        if audio is None:
            audio_path = self.clip_df.iloc[idx]['audio_path']
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
            self.audio_cache[idx] = audio

        return video.to(device), audio.to(device)

    def __len__(self):
        return len(self.clip_df)

    # NOTE: collation ??
