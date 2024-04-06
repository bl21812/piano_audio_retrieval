import os
import torch

from preprocessor import EmbeddingDataset

data_folder = 'data/embeddings/'

model_file = 'runs/t4/best.pt'

num_candidates = 5
device = 'cuda:0'

csv = 'data/val_embeddings.csv'
ds = EmbeddingDataset(
    csv, 
    num_candidates=num_candidates,
    device=device
)
loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)

model = torch.load(model_file).to(device)

for i, (video, audios, label) in enumerate(loader):

    label = label.to(device)

    with torch.no_grad():

        preds = model((video, audios))

        candidate_fnames = ds.current_batch_audios
        print(f'Audio candidates: {candidate_fnames}')
        print(f'Correct choice: {label.item()}')
        print(f'Predicted choice: {torch.max(preds, dim=1)[1].item()}')
        print(f'Output logits (not normalized): {preds}')

    inp = input()
    if not inp:
        break
