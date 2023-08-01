import numpy as np
from mpnn.data import make_batches


class Evaluator:
    def __init__(self, device, model, output_path="", max_n_neigh=None):
        self.device = device
        self.model = model
        self.output_path = output_path
        self.max_n_neigh = max_n_neigh

    def eval(self, input, batch_size):
        # data
        if type(input) is str:
            data = dict(np.load(input, allow_pickle=True))
        elif type(input) is dict:
            data = input
        gen, steps = make_batches(data, batch_size, self.device[0], self.max_n_neigh)

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
            print(f"\r {s / steps * 100: 3f}", end="\r")

        # concatenate results
        for k, v in output.items():
            output[k] = np.concatenate(v, axis=0)

        if self.output_path:
            np.savez_compressed(self.output_path, **output)

        return output
