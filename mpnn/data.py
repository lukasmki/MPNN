import torch
import numpy as np


def batch_conveter(input, device):
    result = {}

    result["R"] = torch.tensor(input["R"], device=device)
    result["Z"] = torch.tensor(input["Z"], dtype=torch.long, device=device)
    result["N"] = torch.tensor(input["N"], dtype=torch.long, device=device)
    result["NM"] = torch.tensor(input["NM"], dtype=torch.long, device=device)
    result["AM"] = torch.tensor(input["AM"], dtype=torch.long, device=device)

    # targets
    for target in ["E", "F", "Ai", "Pij", "Q", "B"]:
        if target not in input:
            continue
        result[target] = torch.tensor(input[target], dtype=torch.float32, device=device)

    return result


def compute_environment(R, Z, max_n_neigh):
    n_data = R.shape[0]
    n_atoms = R.shape[1]

    assert (R.shape[0] == Z.shape[0]) and (R.shape[1] == Z.shape[1])

    N = np.tile(np.arange(n_atoms), (n_atoms, 1))

    neighbors = N[~np.eye(n_atoms, dtype=bool)].reshape(n_atoms, -1)
    neighbors = np.repeat(neighbors[np.newaxis, ...], n_data, axis=0)

    mask = np.ones_like(Z)
    mask[np.where(Z == 0)] = 0
    max_atoms = np.sum(mask, axis=1)

    neighbor_mask = (
        neighbors < np.tile(max_atoms.reshape(-1, 1), n_atoms - 1)[:, None, :]
    ).astype("int")
    neighbor_mask *= mask[:, :, None]
    neighbors *= neighbor_mask

    if (max_n_neigh is not None) and (n_atoms < max_n_neigh):
        neighbors = np.pad(
            neighbors, ((0, 0), (0, 0), (0, max_n_neigh - n_atoms)), constant_values=-1
        )

    return neighbors, neighbor_mask, mask


def batch_generator(data, batch_size, device, max_n_neigh):
    n_data = data["R"].shape[0]
    data_indices = list(range(n_data))
    seen_all_data = False
    while True:
        split = 0
        while (split + 1) * batch_size < n_data:
            batch_indices = data_indices[split * batch_size : (split + 1) * batch_size]
            data_batch = {k: v[batch_indices] for k, v in data.items()}

            N, NM, AM = compute_environment(
                data_batch["R"], data_batch["Z"], max_n_neigh
            )
            data_batch["N"] = N
            data_batch["NM"] = NM
            data_batch["AM"] = AM

            batch = batch_conveter(data_batch, device)
            yield batch
            split += 1

        else:
            batch_indices = data_indices[split * batch_size :]
            data_batch = {k: v[batch_indices] for k, v in data.items()}

            N, NM, AM = compute_environment(
                data_batch["R"], data_batch["Z"], max_n_neigh
            )
            data_batch["N"] = N
            data_batch["NM"] = NM
            data_batch["AM"] = AM

            batch = batch_conveter(data_batch, device)
            yield batch

        seen_all_data = True


def make_batches(data, batch_size, device, max_n_neigh):
    n_data = data["R"].shape[0]
    n_steps = int(np.ceil(n_data / batch_size))
    gen = batch_generator(data, batch_size, device, max_n_neigh)
    return gen, n_steps
