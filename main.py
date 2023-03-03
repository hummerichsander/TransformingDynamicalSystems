import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from dataset import TransformerDataset
from model import TransformerModel


def train(
    fpath: str,
    enc_seq_len: int,
    target_seq_len: int,
    d_obs: int,
    d_model: int,
    num_heads: int,
    enc_num_layers: int,
    dec_num_layers: int,
    num_epochs: int,
    batchsize: int,
    enc_dropout: int = .2,
    dec_dropout: int = .2,
    learning_rate: dict = {0: 1e-3},
) -> None:
    train_dataset = TransformerDataset(fpath, enc_seq_len, target_seq_len)
    train_loader = DataLoader(train_dataset, batchsize)
    enc_mask = train_dataset.enc_mask
    dec_mask = train_dataset.dec_mask

    model = TransformerModel(
        seq_len=enc_seq_len+target_seq_len,
        d_obs=d_obs,
        d_model=d_model,
        num_heads=num_heads,
        enc_num_layers=enc_num_layers,
        dec_num_layers=dec_num_layers,
        enc_dropout=enc_dropout,
        dec_dropout=dec_dropout,
    )

    optimizer = torch.optim.Adam(params=model.parameters())
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_iterator = tqdm(train_loader)
        if epoch in learning_rate.keys():
            optimizer.lr = learning_rate[epoch]
            print("Changed learning rate to:", optimizer.lr)
        for (enc_input, dec_input, target) in epoch_iterator:
            epoch += 1
            loss = 0
            optimizer.zero_grad()

            output = model(enc_input, dec_input, enc_mask, dec_mask)
            loss = loss_fn(output, target)
            epoch_iterator.set_description(f"Loss={loss.item()}")

            loss.backward()
            optimizer.step()

    return model


if __name__ == "__main__":
    model = train(
        "path_to_data",
        enc_seq_len=10,
        target_seq_len=4,
        d_obs=3,
        d_model=512,
        num_heads=8,
        enc_num_layers=4,
        dec_num_layers=4,
        num_epochs=1,
        batchsize=512,
        enc_dropout=0.4,
        dec_dropout=0.4
    )
    torch.save(model, f"./models/model_{datetime.now()}.pt")
