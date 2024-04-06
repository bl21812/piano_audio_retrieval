# GET VIDEO_LLAMA EMBEDDINGS FOR CLIP FRAMES AND AUDIO

import os
import math
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

from models import load_video_llama_modules, embed_clip, embed_audio, AudioRetrievalModel
from preprocessor import VideoAudioDataset

device = 'cuda:0'

ckpt_path = "VL_LLaMA_2_13B_finetuned.pth"
ckpt_2_path = "AL_LLaMA_2_13B_finetuned.pth"

img_dim = 224

split = 'train'
root_dir = 'data/'

clip_embedding_dir = os.path.join(root_dir, 'embeddings', 'clips')
if not os.path.exists(clip_embedding_dir):
    os.makedirs(clip_embedding_dir)
audio_embedding_dir = os.path.join(root_dir, 'embeddings', 'audio')
if not os.path.exists(audio_embedding_dir):
    os.makedirs(audio_embedding_dir)

# video: (batch, channels, time, height, width)
# audios: (batch, num_subclips, (1?), mel_bins, target_length)
ds = VideoAudioDataset(f'data/{split}_ds.csv', device=device, img_dim=img_dim)
loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

# need new dataframes since paths change as well
clip_paths = []
audio_paths = []

# ----- MODEL -----
modules = load_video_llama_modules(img_size=img_dim)
model = AudioRetrievalModel(modules, device='cpu')

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
adjusted_weights = torch.tensor(adjusted_weights, requires_grad=False)
temp_dict = {'video_frame_position_embedding.weight': adjusted_weights}
model.load_state_dict(temp_dict, strict=False)

ckpt = torch.load(ckpt_2_path, map_location="cpu")
model.load_state_dict(ckpt['model'], strict=False)

modules = {
    'visual_encoder': model.visual_encoder,
    'ln_vision': model.ln_vision,
    'Qformer': model.Qformer,
    'query_tokens': model.query_tokens,
    'video_frame_position_embedding': model.video_frame_position_embedding,
    'video_Qformer': model.video_Qformer,
    'video_query_tokens': model.video_query_tokens,
    'audio_encoder': model.audio_encoder,
    'audio_position_embedding': model.audio_position_embedding,
    'audio_Qformer': model.audio_Qformer,
    'audio_query_tokens': model.audio_query_tokens,
}

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
