import sys
sys.path.append('Video-LLaMA')

import torch

import video_llama.models as video_llama

from models import load_video_llama_modules, AudioRetrievalModel, embed_clip, embed_audio
from preprocessor import VideoAudioDataset


device = 'cuda:0'

# video: (batch, channels, time, height, width)
# audios: (batch, num_subclips, (1?), mel_bins, target_length)
train_ds = VideoAudioDataset('data/train_ds.csv', device=device)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)

# ----- MODEL -----
modules = load_video_llama_modules()
print(modules.keys())
print('model loaded')

# ----- TRAIN LOOP -----
for i, (video, audio) in enumerate(train_loader):

    if i == 0:

        print(video.size())
        print(audio.size())
        audio = audio.to('cpu')

        input()

        video_out = embed_clip(video, modules)
        print(video_out.size())
        print('video done')

        video = video.to('cpu')
        audio = audio.to(device)

        audio_out = embed_audio(audio, modules)
        print(audio_out.size())
        print('audio done')

        '''for audio_candidates in audios:
            audio = audio_encoder(aud)
        print('audio done')'''

        input()
        
    else:
        break
