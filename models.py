import sys
sys.path.append('Video-LLaMA')

import contextlib
import einops
import torch
import torch.nn as nn

from video_llama.common.dist_utils import download_cached_file
from video_llama.models.ImageBind.models.imagebind_model import ModalityType
import video_llama.models as video_llama


def disabled_train(self, mode=True):
    return self

# TODO: im not loading their pretrained weights am I? (aren't these just the base ones)
# and vid and audio q former may be randomly initialized rn ??
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
    max_frame_pos=300,
    num_video_query_token = 32,
    num_audio_query_token = 8,
    imagebind_ckpt_path="imagebind_huge.pth",
):

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

    # load q-former from pretrained (hard-coded to download from url here)
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
        'Qformer': Qformer,
        'query_tokens': query_tokens,
        'video_frame_position_embedding': video_frame_position_embedding,
        'video_Qformer': video_Qformer,
        'video_query_tokens': video_query_tokens,
        'audio_encoder': audio_encoder,
        'audio_position_embedding': audio_position_embedding,
        'audio_Qformer': audio_Qformer,
        'audio_query_tokens': audio_query_tokens,
    }


def maybe_autocast(device, dtype=torch.float16):
    # if on cpu, don't use autocast
    # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
    enable_autocast = device != torch.device("cpu")

    if enable_autocast:
        return torch.cuda.amp.autocast(dtype=dtype)
    else:
        return contextlib.nullcontext()


def embed_clip(video, modules, device='cuda:0'):
    '''
    @param video: input of size (batch, channels, time, height, width)
        currently (batch, 3, 300, 224, 224)
    @return video embeddings, of shape (batch, query_tokens, hidden_dim)
        currently (batch, 32, 768)
    '''

    visual_encoder = modules['visual_encoder']
    ln_vision = modules['ln_vision']
    query_tokens = modules['query_tokens']
    Qformer = modules['Qformer']
    video_frame_position_embedding = modules['video_frame_position_embedding']
    video_query_tokens = modules['video_query_tokens']
    video_Qformer = modules['video_Qformer']

    batch_size, _, time_length, _, _ = video.size()
    frames = einops.rearrange(video, 'b c t h w -> (b t) c h w')

    with torch.no_grad():

        with maybe_autocast(device):

            # embed features with blip2
            # query_output shape: ((b * t), q, h)
            visual_encoder = visual_encoder.to(device)
            ln_vision = ln_vision.to(device)
            frame_embeds = ln_vision(visual_encoder(frames))

            visual_encoder = visual_encoder.to('cpu')
            ln_vision = ln_vision.to('cpu')

            query_tokens = query_tokens.to(device)
            Qformer = Qformer.to(device)

            # Qformer
            frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(device)
            query_tokens = query_tokens.expand(frame_embeds.shape[0], -1, -1)
            query_output = Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=frame_embeds,
                encoder_attention_mask=frame_atts,
                return_dict=True,
            )

            query_tokens = query_tokens.to('cpu')
            Qformer = Qformer.to('cpu')

            video_frame_position_embedding = video_frame_position_embedding.to(device)

            # add positional embeddings
            position_ids = torch.arange(time_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = video_frame_position_embedding(position_ids)
            q_hidden_state = query_output.last_hidden_state
            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
            frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h', b=batch_size, t=time_length)
            frame_hidden_state = frame_position_embeddings + frame_hidden_state

            video_frame_position_embedding = video_frame_position_embedding.to('cpu')

            video_query_tokens = video_query_tokens.to(device)
            video_Qformer = video_Qformer.to(device)
            
            # frame attention
            frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length)
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
            video_query_tokens = video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)

            video_query_output = video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            video_hidden = video_query_output.last_hidden_state

            video_query_tokens = video_query_tokens.to('cpu')
            video_Qformer = video_Qformer.to('cpu')

    return video_hidden


def embed_audio(audio, modules, device='cuda:0'):
    '''
    @param audio: input of shape (batch, num_subclips, (1?), mel_bins, target_length)
        currently (batch, 5, 1, 128, 204)
    @return audio embeddings, of shape (batch, query_tokens, hidden_dim)
        currently (batch, 8, 768)
    '''

    audio_encoder = modules['audio_encoder']
    audio_position_embedding = modules['audio_position_embedding']
    audio_Qformer = modules['audio_Qformer']
    audio_query_tokens = modules['audio_query_tokens']

    with torch.no_grad():

        with maybe_autocast(device):

            audio_encoder = audio_encoder.to(device)

            # encode with ImageBind
            audio_feature, audio_imagebind_finalout = audio_encoder.get_audio_feature(audio, modality_type=ModalityType.AUDIO)
            batch_size, time_length = audio.size()[:2]
            
            audio_encoder = audio_encoder.to('cpu')

            audio_position_embedding = audio_position_embedding.to(device)

            # position embedding
            position_ids = torch.arange(time_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            audio_position_embeddings = audio_position_embedding(position_ids)
            audio_imagebind_finalout = audio_imagebind_finalout + audio_position_embeddings

            audio_position_embedding = audio_position_embedding.to('cpu')

            audio_query_tokens = audio_query_tokens.to(device)
            audio_Qformer = audio_Qformer.to(device)

            # Qformer
            audio_query_tokens = audio_query_tokens.expand(audio_imagebind_finalout.shape[0], -1, -1)
            frame_atts = torch.ones(audio_imagebind_finalout.size()[:-1], dtype=torch.long).to(device)

            audio_query_output = audio_Qformer.bert(
                query_embeds=audio_query_tokens, #[32,768]
                encoder_hidden_states=audio_imagebind_finalout,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )

            audio_hidden = audio_query_output.last_hidden_state

            audio_query_tokens = audio_query_tokens.to('cpu')
            audio_Qformer = audio_Qformer.to('cpu')

    return audio_hidden


class AudioRetrievalHead(nn.Module):

    def __init__(
        self, 
        attention_dim=768,
        attention_heads=8,
        input_dim=(32*768), 
        num_hidden=2, 
        hidden_dim=64, 
        num_candidates=5, 
        device='cuda:0'
    ):

        assert num_hidden > 1, 'Does not currently support no hidden layers!'

        super().__init__()

        # attention layer
        # outputs (32, 768) tensor 
        self.attention = nn.MultiheadAttention(
            attention_dim, 
            attention_heads, 
            batch_first=True
        ).to(device)

        # FC layers
        '''layers = []
        in_dim = input_dim
        out_dim = hidden_dim
        for i in range(num_hidden):
            if i > 0:
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim, device=device))
            layers.append(nn.ReLU())
        self.fc_layers = nn.Sequential(*layers).to(device)'''
        layers = []
        for i in range(num_hidden):
            layers.append(nn.MultiheadAttention(
                attention_dim, 
                attention_heads,
                batch_first=True
            ).to(device))
        self.self_attention = layers

        # output layer
        # in_dim = in_dim * num_candidates
        in_dim = attention_dim * 32 * num_candidates
        out_dim = num_candidates
        self.output_layer = nn.Linear(in_dim, out_dim, device=device)

    def forward(self, x):
        '''
        For each video-audio pair, applies MHA and FC layers (shared params)
            and intermediate tensors are concatenated for final output layer

        @param x: tuple of (clip, audio_candidates)
            clip: clip embedding, size (batch, video_query_tokens, hidden_dim)
            audio_candidates: audio embeddings,
                of size (batch, candidates, audio_query_tokens, hidden_dim)
        '''

        clip, audio_candidates = x

        # change shape to (num_candidates, batch, ...)
        audio_candidates = torch.transpose(audio_candidates, 0, 1)  

        # apply MHA and FC layers
        fc_outs = []
        for audio_candidate in audio_candidates:

            att_out = self.attention(
                query=clip,
                key=audio_candidate,
                value=audio_candidate,
                need_weights=False
            )[0]
            # NOTE: consider averaging over each seq_len dim (remove the 32 dim) instead?

            for att_layer in self.self_attention:
                att_out = att_layer(
                    query=att_out,
                    key=att_out,
                    value=att_out,
                    need_weights=False
                )[0]

            att_out = torch.flatten(att_out, start_dim=1)  # keep batch dim

            fc_outs.append(att_out)

            # fc_out = self.fc_layers(att_out)
            # fc_outs.append(fc_out)

        # concat and output
        fc_cat = torch.cat(fc_outs, dim=1)
        outs = self.output_layer(fc_cat)
        
        return outs


# ----- NOT USING -----

# NOTE: this currently must all run without grads
# (due to memory / device-torch compatibility restrictions)
# im not using this - but the code is the same above
class AudioRetrievalModel(nn.Module):

    def __init__(self, video_llama_modules, device='cuda:0'):

        super().__init__()
        self.device = device

        self.visual_encoder = video_llama_modules['visual_encoder'].to(device)
        self.ln_vision = video_llama_modules['ln_vision'].to(device)
        self.Qformer = video_llama_modules['Qformer'].to(device)
        self.query_tokens = video_llama_modules['query_tokens'].to(device)
        self.video_frame_position_embedding = video_llama_modules['video_frame_position_embedding'].to(device)
        self.video_Qformer = video_llama_modules['video_Qformer'].to(device)
        self.video_query_tokens = video_llama_modules['video_query_tokens'].to(device)

        self.audio_encoder = video_llama_modules['audio_encoder'].to(device)
        self.audio_position_embedding = video_llama_modules['audio_position_embedding'].to(device)
        self.audio_Qformer = video_llama_modules['audio_Qformer'].to(device)
        self.audio_query_tokens = video_llama_modules['audio_query_tokens'].to(device)

    def video_to_device(self):
        self.visual_encoder = self.visual_encoder.to(self.device)
        self.ln_vision = self.ln_vision.to(self.device)
        self.Qformer = self.Qformer.to(self.device)
        self.query_tokens = self.query_tokens.to(self.device)
        self.video_frame_position_embedding = self.video_frame_position_embedding.to(self.device)
        self.video_Qformer = self.video_Qformer.to(self.device)
        self.video_query_tokens = self.video_query_tokens.to(self.device)

    def video_to_cpu(self):
        self.visual_encoder = self.visual_encoder.to('cpu')
        self.ln_vision = self.ln_vision.to('cpu')
        self.Qformer = self.Qformer.to('cpu')
        self.query_tokens = self.query_tokens.to('cpu')
        self.video_frame_position_embedding = self.video_frame_position_embedding.to('cpu')
        self.video_Qformer = self.video_Qformer.to('cpu')
        self.video_query_tokens = self.video_query_tokens.to('cpu')

    def audio_to_device(self):
        self.audio_encoder = self.audio_encoder.to(self.device)
        self.audio_position_embedding = self.audio_position_embedding.to(self.device)
        self.audio_Qformer = self.audio_Qformer.to(self.device)
        self.audio_query_tokens = self.audio_query_tokens.to(self.device)

    def audio_to_cpu(self):
        self.audio_encoder = self.audio_encoder.to('cpu')
        self.audio_position_embedding = self.audio_position_embedding.to('cpu')
        self.audio_Qformer = self.audio_Qformer.to('cpu')
        self.audio_query_tokens = self.audio_query_tokens.to('cpu')

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def forward_video(self, video):
        '''
        video: batch of videos, of shape (b, c, t, h, w)
        return: learned video embedding, of shape ()
        '''

        device = video.device

        batch_size, _, time_length, _, _ = video.size()
        frames = einops.rearrange(video, 'b c t h w -> (b t) c h w')

        with torch.no_grad():

            with self.maybe_autocast():

                self.visual_encoder = self.visual_encoder.to(device)
                self.ln_vision = self.ln_vision.to(device)

                # embed features with blip2
                # query_output shape: ((b * t), q, h)
                frame_embeds = self.ln_vision(self.visual_encoder(frames))
                frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(device)

                # TODO: try deleting with 'del' to clear memory
                self.visual_encoder = self.visual_encoder.to('cpu')
                self.ln_vision = self.ln_vision.to('cpu')
                print(self.query_tokens)
                input()
                self.query_tokens = self.query_tokens.to(device)
                self.Qformer = self.Qformer.to(device)

                query_tokens = self.query_tokens.expand(frame_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=frame_embeds,
                    encoder_attention_mask=frame_atts,
                    return_dict=True,
                )

                self.query_tokens = self.query_tokens.to('cpu')
                self.Qformer = self.Qformer.to('cpu')
                self.video_frame_position_embedding = self.video_frame_position_embedding.to(device)

                # add positional embeddings
                position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                frame_position_embeddings = self.video_frame_position_embedding(position_ids)
                q_hidden_state = query_output.last_hidden_state

                self.video_frame_position_embedding = self.video_frame_position_embedding.to('cpu')

                frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
                frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h', b=batch_size, t=time_length)
                frame_hidden_state = frame_position_embeddings + frame_hidden_state

                self.video_query_tokens = self.video_query_tokens.to(device)
                self.video_Qformer = self.video_Qformer.to(device)

                # frame attention
                frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length)
                frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
                video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)

                video_query_output = self.video_Qformer.bert(
                    query_embeds=video_query_tokens,
                    encoder_hidden_states=frame_hidden_state,
                    encoder_attention_mask=frame_atts,
                    return_dict=True,
                    )
                video_hidden = video_query_output.last_hidden_state

                self.video_query_tokens = self.video_query_tokens.to('cpu')
                self.video_Qformer = self.video_Qformer.to('cpu')

        return video_hidden

    def forward_audio(self, audio):
        '''
        audio: batch of groups of audio, list of b elements of shapes (n, ___)
            where n is number of candidate audios per video
        return: learned audio embeddings, of shape (b, ___)
        '''

        device = audio[0].device
        audio_embeddings = []

        for aud in audio:

            with torch.no_grad():

                with self.maybe_autocast():

                    audio_feature, audio_imagebind_finalout = self.audio_encoder.get_audio_feature(aud, modality_type=ModalityType.AUDIO)
                    batch_size, time_length = aud.size()[:2]

                    position_ids = torch.arange(time_length, dtype=torch.long, device=device)
                    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

                    audio_position_embeddings = self.audio_position_embedding(position_ids)
                    audio_imagebind_finalout = audio_imagebind_finalout + audio_position_embeddings

                    audio_query_tokens = self.audio_query_tokens.expand(audio_imagebind_finalout.shape[0], -1, -1)
                    frame_atts = torch.ones(audio_imagebind_finalout.size()[:-1], dtype=torch.long).to(device)

                    audio_query_output = self.audio_Qformer.bert(
                        query_embeds=audio_query_tokens, #[32,768]
                        encoder_hidden_states=audio_imagebind_finalout,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                        )
                    audio_hidden = audio_query_output.last_hidden_state

                    audio_embeddings.append(audio_hidden)

        return torch.stack(audio_embeddings, dim=0)

    def forward(self, x):
        '''
        x: 
        '''
        # encode video and all audios
        # return each video embedding w/ corresponding audio candidate embeddings
        return x
