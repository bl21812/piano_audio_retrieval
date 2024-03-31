import sys
sys.path.append('Video-LLaMA')

import torch

import video_llama.models as video_llama

from models import load_video_llama_modules, AudioRetrievalModel
from preprocessor import VideoAudioDataset


device = 'cuda:0'

# video: (batch, channels, time, height, width)
# audios: (batch, candidates, num_subclips, (1?), mel_bins, target_length)
train_ds = VideoAudioDataset('data/train_ds.csv', device=device)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)

# ----- MODEL -----
modules = load_video_llama_modules()
print(modules.keys())

model = AudioRetrievalModel(modules, device=device)
print('model loaded')

# ----- TRAIN LOOP -----
for i, (videos, audios) in enumerate(train_loader):

    if i == 0:

        print(videos.size())
        print(audios.size())

        video_out = model.forward_video(videos)
        print(video_out.size())
        print('video done')

        '''for audio_candidates in audios:
            audio = audio_encoder(aud)
        print('audio done')'''

        input()
        
    else:
        break
