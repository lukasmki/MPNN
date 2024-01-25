import numpy as np
from mpnn.data import make_batches
from torch import nn


class Evaluator:
    def __init__(self, device, model, output_path="", max_n_neigh=None):
        self.device = device
        self.model = model
        self.output_path = output_path
        self.max_n_neigh = max_n_neigh
        self.parallelized = False

        if isinstance(device, list) and len(device) > 1:
            self.multi_gpu = True
        else:
            self.multi_gpu = False

    def eval(self, input, batch_size=1):
        # data
        if isinstance(input, str):
            data = dict(np.load(input, allow_pickle=True))
        elif isinstance(input, dict):
            data = input
        gen, steps = make_batches(data, batch_size, self.device[0], self.max_n_neigh)

        # move model to device
        self.model.to(self.device[0])
        if (self.multi_gpu) and not self.parallelized:
            self.model = nn.DataParallel(self.model, device_ids=self.device)
            self.parallelized = True

        # iterate over batches
        output = {}
        for s in range(steps):
            batch = next(gen)
            preds = self.model(batch)

            # storing results
            for k, v in preds.items():
                if k in output:
                    output[k].append(v.detach().cpu().numpy())
                else:
                    output[k] = [v.detach().cpu().numpy()]

            # output
            print(f"\r running...{s / steps:3.2%}", end="\r")
            del batch

        # concatenate results
        shapes = np.array([list(x.shape) for x in output["N"]])
        nframes = np.sum(shapes[::, 0])
        natoms = np.max(shapes[::, 1])
        nneigh = np.max(shapes[::, 2])

        for k, v in output.items():
            # for arr in output[k]:
            #     print(arr.shape)

            output[k] = np.concatenate(output[k], axis=0)

        if self.output_path:
            np.savez_compressed(self.output_path, **output)

        return output
