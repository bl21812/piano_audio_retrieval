# GET VIDEO_LLAMA EMBEDDINGS FOR CLIP FRAMES AND AUDIO

import torch
import tqdm
import numpy as np

from models import load_video_llama_modules, embed_clip, embed_audio
from preprocessor import VideoAudioDataset

device = 'cuda:0'

split = 'train'

# video: (batch, channels, time, height, width)
# audios: (batch, num_subclips, (1?), mel_bins, target_length)
ds = VideoAudioDataset(f'data/{split}_ds.csv', device=device)
loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

# ----- MODEL -----
modules = load_video_llama_modules()
print(modules.keys())
print('model loaded')

# embed everything !
video_embeddings = []
audio_embeddings = []

video_embedding_file = f'{split}_clip_embeddings.npy'
audio_embedding_file = f'{split}_audio_embeddings.npy'

for i, (video, audio) in enumerate(tqdm(loader)):

    audio = audio.to('cpu')

    # embed video
    video_out = embed_clip(video, modules)
    video_out = np.squeeze(video_out.to('cpu').numpy())

    video = video.to('cpu')
    audio = audio.to(device)

    # embed audio
    audio_out = embed_audio(audio, modules)
    audio_out = np.squeeze(audio_out.to('cpu').numpy())

    # NOTE: may have to switch to writing to file instead of appending based on memory !
    video_embeddings.append(video_out.to('cpu'))
    audio_embeddings.append(audio_out.to('cpu'))

video_embeddings = np.stack(video_embeddings)
np.save(video_embedding_file, video_embeddings)

audio_embeddings = np.stack(audio_embeddings)
np.save(audio_embedding_file, audio_embeddings)
