import sys
sys.path.append('Video-LLaMA')

import pandas as pd
import numpy as np
import random

import decord
from decord import VideoReader

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from video_llama.models.ImageBind.data import load_and_transform_audio_data
from video_llama.processors.video_processor import load_video, ToTHWC, ToUint8
from video_llama.processors import transforms_video

IMG_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMG_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_SIZE = 480


class EmbeddingDataset(Dataset):

    def __init__(self, embedding_file, num_candidates=5, device='cuda:0'):
        self.df = pd.read_csv(embedding_file)
        self.device = device
        self.num_candidates = num_candidates
        self.current_batch_audios = None

    def __getitem__(self, idx):
        '''
        return: (video, audio_candidates, label)
            video: shape (num_queries, hidden_dim)
                currently (32, 768)
            audio_candidates: shape (num_candidates, num_queries, hidden_dim)
                currently (5, 8, 768)
            label: integer index corresponding to true audio
        '''

        row = self.df.iloc[idx]

        video = np.load(row['clip'])
        video = torch.tensor(video).to(self.device)

        # choose candidates and shuffle index order
        candidate_idxs = np.random.randint(low=0, high=len(self.df)-1, size=(self.num_candidates-1)).tolist()
        for i, candidate_idx in enumerate(candidate_idxs):
            if candidate_idx >= idx:
                candidate_idxs[i] += 1
        
        candidate_idx_iter = iter(candidate_idxs)
        gt_idx = np.random.randint(low=0, high=self.num_candidates)
        audio_idxs = []
        for i in range(self.num_candidates):
            if i == gt_idx:
                audio_idxs.append(idx)
            else:
                audio_idxs.append(next(candidate_idx_iter))
    
        # Load all candidate audios
        audios = []
        for audio_idx in audio_idxs:
            audio = np.load(self.df.iloc[audio_idx]['audio'])
            audio = torch.tensor(audio).to(self.device)
            audios.append(audio)

        # build classification target
        # label = np.zeros(self.num_candidates)
        # label[gt_idx] = 1.0

        # for demo purposes
        candidate_fnames = []
        for audio_idx in audio_idxs:
            candidate_fnames.append(((self.df.iloc[audio_idx]['audio'].split('/'))[-1]).split('.')[0])
        self.current_batch_audios = candidate_fnames

        return video, torch.stack(audios), gt_idx

    def __len__(self):
        return len(self.df)


# NOTE: this just returns a single video, audio pair
    # intended for preprocessing
class VideoAudioDataset(Dataset):

    def __init__(self, dataset_csv, num_candidates=5, device='cuda:0', img_dim=224):

        self.device = device
        self.num_candidates = num_candidates
        self.img_dim = img_dim

        self.clip_df = pd.read_csv(dataset_csv)
        print(f'Loaded dataframe with {len(self.clip_df)} entries')

        self.video_pipeline = transforms.Compose([
            ToTHWC(),
            ToUint8(),
            transforms_video.ToTensorVideo(),
            transforms_video.NormalizeVideo(IMG_MEAN, IMG_STD)
        ])

    # return video tensor, plus set of candidate audios (random)
        # where first candidate audio is ground truth
    # assuming all video/audio is already cached!
    def __getitem__(self, idx):

        # load video tensor
        clip_path = self.clip_df.iloc[idx]['clip_path']
        # leaving n_frms as MAX_INT in the below
        video = load_video(
            clip_path,
            n_frms=1000000,
            height=self.img_dim,
            width=self.img_dim,
            sampling='uniform',
            return_msg=False  # True to display number of frames and sampling interval
        )
        video = self.video_pipeline(video).to(self.device)
        # shape = (c, t, h, w)

        # load audio tensor
        audio_path = self.clip_df.iloc[idx]['audio_path']
        audio = load_and_transform_audio_data(
            [audio_path], 
            device='cpu',
            num_mel_bins=128, 
            target_length=204,
            sample_rate=16000,
            clip_duration=2,
            clips_per_video=5,  # to use all audio in each 10 sec clip
            mean=-4.268,
            std=9.138,
        )
        audio = torch.squeeze(audio, 0).to(self.device)  # remove leading dim because only using 1 input

        filename = ((self.clip_df.iloc[idx]['clip_path'].split('/'))[-1]).split('.')[0]
        
        return video, audio, filename

    def __len__(self):
        return len(self.clip_df)

    def get_filename(self, idx):
        return self.clip_df.iloc[idx]
