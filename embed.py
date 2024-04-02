# GET VIDEO_LLAMA EMBEDDINGS FOR CLIP FRAMES AND AUDIO

import os
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

from models import load_video_llama_modules, embed_clip, embed_audio
from preprocessor import VideoAudioDataset

device = 'cuda:0'

split = 'val'
root_dir = 'data/'

clip_embedding_dir = os.path.join(root_dir, 'embeddings', 'clips')
if not os.path.exists(clip_embedding_dir):
    os.makedirs(clip_embedding_dir)
audio_embedding_dir = os.path.join(root_dir, 'embeddings', 'audio')
if not os.path.exists(audio_embedding_dir):
    os.makedirs(audio_embedding_dir)

# video: (batch, channels, time, height, width)
# audios: (batch, num_subclips, (1?), mel_bins, target_length)
ds = VideoAudioDataset(f'data/{split}_ds.csv', device=device)
loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

# need new dataframes since paths change as well
clip_paths = []
audio_paths = []

# ----- MODEL -----
modules = load_video_llama_modules()
print(modules.keys())
print('model loaded')

# embed everything !
video_embeddings = []
audio_embeddings = []

for i, (video, audio, filename) in enumerate(tqdm(loader)):

    filename = filename[0]

    audio = audio.to('cpu')

    # embed video
    video_out = embed_clip(video, modules)
    video_out = np.squeeze(video_out.to('cpu').numpy())

    video = video.to('cpu')
    audio = audio.to(device)

    # embed audio
    audio_out = embed_audio(audio, modules)
    audio_out = np.squeeze(audio_out.to('cpu').numpy())

    # write to npy file
    video_out_path = os.path.join(clip_embedding_dir, f'{filename}.npy')
    clip_paths.append(video_out_path)
    
    audio_out_path = os.path.join(audio_embedding_dir, f'{filename}.npy')
    audio_paths.append(audio_out_path)

    np.save(video_out_path, video_out)
    np.save(audio_out_path, audio_out)

# write clip and audio paths to df
clip_df_path = os.path.join(root_dir, f'{split}_embeddings.csv')
clip_df = pd.DataFrame({'clip': clip_paths, 'audio': audio_paths})
clip_df.to_csv(clip_df_path, index=False)
