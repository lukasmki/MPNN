import os
import pathlib
import numpy as np
import torch
from torch.optim.adam import Adam
from mpnn import Trainer, Evaluator
from mpnn.models import NewtonNet

# fake training data
trn_data = {
    "R": np.random.uniform(0, 1, (100, 6, 3)),
    "Z": np.repeat([[1, 1, 8, 1, 1, 8]], 100, axis=0),
    "N": np.repeat([[6]], 100, axis=0),
}
trn_data.update(
    {
        "E": np.sum(trn_data["R"], (1, 2)),
        "F": np.random.normal(0, 1, (100, 6, 3)),
    }
)

ROOT_DIR = pathlib.Path(__file__).parent.resolve()

device = [torch.device(x) for x in ["cpu"]]
model = NewtonNet(device=device[0])

trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = Adam(trainable_params, lr=0.0001, weight_decay=0)


def loss(epoch, preds, batch_data):
    diff_en = preds["E"] - batch_data["E"]
    err_en = torch.mean(diff_en * diff_en)

    diff_fc = preds["F"] - batch_data["F"]
    err_fc = 20 * torch.mean(diff_fc * diff_fc)

    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    diff_fd = 1 - cos(preds["F_latent"], batch_data["F"])
    err_fd = torch.mean(diff_fd)

    err_sq = err_en + err_fc + err_fd

    return err_sq


trainer = Trainer(
    device,
    model,
    loss,
    optimizer,
    os.path.join(ROOT_DIR, "model"),
    ["plateau", 5, 10, 0.7, 1.0e-6],
)

trainer.train(
    epochs=10,
    training_set=trn_data,
    train_batch_size=5,
)

# fake test data
tst_data = {
    "R": np.random.uniform(0, 1, (100, 6, 3)),
    "Z": np.repeat([[1, 1, 8, 1, 1, 8]], 100, axis=0),
    "N": np.repeat([[6]], 100, axis=0),
}
run = Evaluator(
    device,
    model,
    os.path.join(ROOT_DIR, "eval"),
)

run.eval(tst_data, batch_size=10)
