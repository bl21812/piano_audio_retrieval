import sys
sys.path.append('Video-LLaMA')

import torch
import torch.nn as nn

from video_llama.common.dist_utils import download_cached_file
import video_llama.models as video_llama


def disabled_train(self, mode=True):
    return self


def load_video_llama_modules(
    vit_model="eva_clip_g",
    q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
    img_size=224,
    drop_path_rate=0,
    use_grad_checkpoint=False,
    vit_precision="fp16",
    freeze_vit=True,
    freeze_qformer=True,
    frozen_video_Qformer=True,
    frozen_audio_Qformer=True,
    num_query_token=32,
    max_frame_pos= 32,
    num_video_query_token = 32,
    num_audio_query_token = 8,
    imagebind_ckpt_path="imagebind_huge.pth",
):

    # load tokenizer?
    # what is low resource?

    # ----- LOAD VIT (img encoder) -----
    # ln = layer norm
    visual_encoder, ln_vision = video_llama.Blip2Base.init_vision_encoder(
        vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
    )

    if freeze_vit:

        for name, param in visual_encoder.named_parameters():
            param.requires_grad = False
        visual_encoder = visual_encoder.eval()
        visual_encoder.train = disabled_train

        for name, param in ln_vision.named_parameters():
            param.requires_grad = False
        ln_vision = ln_vision.eval()
        ln_vision.train = disabled_train

        print("freeze vision encoder")

    print("VIT loaded")

    # ----- LOAD Q-FORMER -----
    Qformer, query_tokens = video_llama.Blip2Base.init_Qformer(
        num_query_token, visual_encoder.num_features
    )

    Qformer.cls = None
    Qformer.bert.embeddings.word_embeddings = None
    Qformer.bert.embeddings.position_embeddings = None
    for layer in Qformer.bert.encoder.layer:
        layer.output = None
        layer.intermediate = None

    # load from pretrained (hard-coded to download from url here)
    # ???
    cached_file = download_cached_file(q_former_model, check_hash=False, progress=True)
    checkpoint = torch.load(cached_file, map_location="cpu")
    state_dict = checkpoint['model']
    Qformer.load_state_dict(state_dict, strict=False)

    if freeze_qformer:
        for name, param in Qformer.named_parameters():
            param.requires_grad = False
        Qformer = Qformer.eval()
        Qformer.train = disabled_train
        query_tokens.requires_grad = False
        print("freeze Qformer")

    print('Q-former loaded')

    # ----- VIDEO Q-FORMER -----
    video_frame_position_embedding = nn.Embedding(max_frame_pos, Qformer.config.hidden_size)

    video_Qformer, video_query_tokens = video_llama.VideoLLAMA.init_video_Qformer(
        num_query_token=num_video_query_token, 
        vision_width=Qformer.config.hidden_size,
        num_hidden_layers=2
    )

    video_Qformer.cls = None
    video_Qformer.bert.embeddings.word_embeddings = None
    video_Qformer.bert.embeddings.position_embeddings = None
    for layer in video_Qformer.bert.encoder.layer:
        layer.output = None
        layer.intermediate = None

    if frozen_video_Qformer:
        for name, param in video_Qformer.named_parameters():
            param.requires_grad = False
        for name, param in video_frame_position_embedding.named_parameters():
            param.requires_grad = False
        video_query_tokens.requires_grad = False
        print('video_Qformer is frozen')
    else:
        for name, param in video_Qformer.named_parameters():
            param.requires_grad = True
        for name, param in video_frame_position_embedding.named_parameters():
            param.requires_grad = True
        video_query_tokens.requires_grad = True
        print('video_Qformer is not frozen')

    # ----- AUDIO Q-FORMER -----
    audio_encoder, audio_hidden_size = video_llama.ImageBind.models.imagebind_model.imagebind_huge()
    audio_encoder.load_state_dict(torch.load(imagebind_ckpt_path))

    # free vision encoder
    for name, param in audio_encoder.named_parameters():
        param.requires_grad = False
    audio_encoder.eval()

    audio_Qformer, audio_query_tokens = video_llama.VideoLLAMA.init_video_Qformer(
        num_query_token=num_audio_query_token,
        vision_width=audio_hidden_size, 
        num_hidden_layers=2
    )
    audio_Qformer.cls = None
    audio_Qformer.bert.embeddings.word_embeddings = None
    audio_Qformer.bert.embeddings.position_embeddings = None
    for layer in audio_Qformer.bert.encoder.layer:
        layer.output = None
        layer.intermediate = None
    
    audio_position_embedding = nn.Embedding(8, audio_hidden_size)

    if frozen_audio_Qformer:
        for name, param in audio_Qformer.named_parameters():
            param.requires_grad = False
        audio_query_tokens.requires_grad = False
        for name, param in audio_position_embedding.named_parameters():
            param.requires_grad = False
        print('audio_Qformer is frozen')
    else:
        for name, param in audio_Qformer.named_parameters():
            param.requires_grad = True
        audio_query_tokens.requires_grad = True
        for name, param in audio_position_embedding.named_parameters():
            param.requires_grad = True
        print('audio_Qformer is not frozen')

    return {
        'visual_encoder': visual_encoder,
        'ln_vision': ln_vision,
        'video_frame_position_embedding': video_frame_position_embedding,
        'video_Qformer': video_Qformer,
        'video_query_tokens': video_query_tokens,
        'audio_encoder': audio_encoder,
        'audio_position_embedding': audio_position_embedding,
        'audio_Qformer': audio_Qformer,
        'audio_query_tokens': audio_query_tokens,
    }
