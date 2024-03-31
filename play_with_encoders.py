import sys
sys.path.append('Video-LLaMA')

import torch

import video_llama.models as video_llama

from models import load_video_llama_modules
from preprocessor import VideoAudioDataset

# zoo = models.ModelZoo()
# print(zoo)

modules = load_video_llama_modules()

'''for module_name, module in modules.items():
    print(module)
    print(module_name)
    input()'''

# audio preprocessing - take spectrograms of consecutive 2-sec audio clips
    # as basically a time series of images

train_ds = VideoAudioDataset('data/train_ds.csv')
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)

for i, (video, audios) in enumerate(train_loader):
    if i == 0:
        print(video.size())
        print(audios.size())
    else:
        break
