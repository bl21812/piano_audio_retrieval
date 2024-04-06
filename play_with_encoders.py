import sys
sys.path.append('Video-LLaMA')

import torch
import math
import numpy as np

from video_llama.common.config import Config
from video_llama.common.registry import registry
import video_llama.models as video_llama

from models import load_video_llama_modules, AudioRetrievalModel, embed_clip, embed_audio
from preprocessor import VideoAudioDataset


device = 'cuda:0'

ckpt_path = "VL_LLaMA_2_13B_finetuned.pth"
ckpt_2_path = "AL_LLaMA_2_13B_finetuned.pth"

modules = load_video_llama_modules()

model = AudioRetrievalModel(modules, device=device)

ckpt = torch.load(ckpt_path, map_location="cpu")
params = ckpt['model']
# shape: 32 x 768
pretrained_embedding_weights = params['video_frame_position_embedding.weight'].numpy()
del params['video_frame_position_embedding.weight']
model.load_state_dict(ckpt['model'], strict=False)

# load video frame position embedding
dummy_layer = torch.nn.Embedding(32, 768)
scale_factor = math.ceil(300 / 32)
adjusted_weights = np.zeros((300, 768))
for i in range(300):
    old_idx = int(i / scale_factor)
    adjusted_weights[i,:] = pretrained_embedding_weights[old_idx,:]
adjusted_weights = torch.tensor(adjusted_weights, requires_grad=False).to(device)
temp_dict = {'video_frame_position_embedding.weight': adjusted_weights}
model.load_state_dict(temp_dict, strict=False)

ckpt = torch.load(ckpt_2_path, map_location="cpu")
model.load_state_dict(ckpt['model'], strict=False)

print('Model weights loaded')

quit()





class Args:

    def __init__(self):
        return

arg_params = {
    "cfg_path": "video_llama_cfg.yaml", 
    "gpu_id": 0,
    "model_type": "vicuna",
    "options": None
}

args = Args()
args.cfg_path = arg_params['cfg_path']
args.gpu_id = arg_params['gpu_id']
args.model_type = arg_params['model_type']
args.options = arg_params['options']

cfg = Config(args)
print(cfg)

model_config = cfg.model_cfg
print(model_config)

model_cls = registry.get_model_class(model_config.arch)
print(model_cls)

model = model_cls.from_config(model_config)
print(model)

# video: (batch, channels, time, height, width)
# audios: (batch, num_subclips, (1?), mel_bins, target_length)
'''train_ds = VideoAudioDataset('data/train_ds.csv', device=device)
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

        input()
        
    else:
        break'''
