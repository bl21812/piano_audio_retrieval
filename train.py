import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocessor import EmbeddingDataset
from models import AudioRetrievalHead

num_candidates = 5
device = 'cuda:0'

train_csv = 'data/train_embeddings.csv'
val_csv = 'data/val_embeddings.csv'

save_dir = 'runs/t4/'

# ----- HPARAMS -----
attention_heads = 64
num_hidden = 5
hidden_dim = 1024  # ONLY WHEN USING LINEAR HIDDENS
lr = 0.001
batch_size = 32
epochs = 50

# ----- LOAD DATASETS -----
train_ds = EmbeddingDataset(
    train_csv, 
    num_candidates=num_candidates,
    device=device
)

val_ds = EmbeddingDataset(
    val_csv, 
    num_candidates=num_candidates,
    device=device
)

# video: (batch_size, 32, 768)
# audio: (batch_size, num_candidates, 8, 768)
# label: (batch_size,)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# ----- MODEL -----
model = AudioRetrievalHead(
    attention_heads=attention_heads,
    num_hidden=num_hidden,
    hidden_dim=hidden_dim
)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ----- TRAIN LOOP -----
best_acc = 0
best_model = None
best_epoch = 0

epoch_train_losses = []
epoch_val_losses = []
epoch_val_acc = []

for epoch in range(epochs):

    epoch_train_loss = 0

    # train
    for i, (video, audios, label) in enumerate(tqdm(train_loader)):

        optimizer.zero_grad()

        preds = model((video, audios))
        l = loss(preds, label.to(device))

        l.backward()
        optimizer.step()

        epoch_train_loss += l.detach().cpu().numpy()

    epoch_train_loss /= len(train_loader)
    epoch_train_losses.append(epoch_train_loss)
    print(f'Finished epoch {epoch}, with train loss: {epoch_train_loss}')

    epoch_val_loss = 0
    total = 0
    correct = 0

    # val
    for i, (video, audios, label) in enumerate(tqdm(val_loader)):

        with torch.no_grad():

            label = label.to(device)

            preds = model((video, audios))
            l = loss(preds, label)

            epoch_val_loss += l.detach().cpu().numpy()
            
            _, pred_classes = torch.max(preds, dim=1)
            total += label.size(0)
            correct += (label == pred_classes).sum().item()

    epoch_val_loss /= len(val_loader)
    epoch_val_losses.append(epoch_val_loss)
    print(f'Epoch {epoch} val loss: {epoch_val_loss}')

    acc = correct / total
    epoch_val_acc.append(acc)
    print(f'Epoch {epoch} accuracy: {acc}')

    if acc > best_acc:
        best_acc = acc
        best_model = copy.deepcopy(model)
        best_epoch = epoch

print()
print('Model training finished')
print(f'Best epoch was epoch {best_epoch}')
print(f'With an accuracy of {best_acc}')
print(f'And a val loss of {epoch_val_losses[best_epoch]}')
print()

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# save best model
torch.save(best_model, os.path.join(save_dir, 'best.pt'))

# save plots
plt.plot([i for i in range(epochs)], epoch_train_losses, color='b', label='train')
plt.plot([i for i in range(epochs)], epoch_val_losses, color='r', label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.savefig(os.path.join(save_dir, 'loss.png'))

plt.clf()
plt.plot([i for i in range(epochs)], epoch_val_acc)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig(os.path.join(save_dir, 'accuracy.png'))

np.save(os.path.join(save_dir, 'train_loss.npy'), np.array(epoch_train_losses))
np.save(os.path.join(save_dir, 'val_loss.npy'), np.array(epoch_val_losses))
np.save(os.path.join(save_dir, 'val_acc.npy'), np.array(epoch_val_acc))
