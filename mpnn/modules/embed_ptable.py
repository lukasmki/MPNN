import torch
from torch import nn
from matplotlib import pyplot as plt


class ptable_embedding(nn.Module):
    def __init__(self, n_features=2):
        super(ptable_embedding, self).__init__()
        self.row_embedding = nn.Embedding(7, n_features, padding_idx=0, max_norm=1)
        self.col_embedding = nn.Embedding(32, n_features, padding_idx=0, max_norm=1)

        # periodic table
        table = torch.zeros((7, 32), dtype=int)
        table[0, 0] = 1
        table[0, 31] = 2

        table[1, :2] = torch.LongTensor([3, 4])
        table[1, 26:] = torch.LongTensor([5, 6, 7, 8, 9, 10])

        table[2, :2] = torch.LongTensor([11, 12])
        table[2, 26:] = torch.LongTensor([13, 14, 15, 16, 17, 18])

        table[3, :2] = torch.LongTensor([19, 20])
        table[3, 16:] = torch.LongTensor(range(21, 37))

        table[4, :2] = torch.LongTensor([37, 38])
        table[4, 16:] = torch.LongTensor(range(39, 55))

        table[5, ::] = torch.LongTensor(range(55, 87))
        table[6, ::] = torch.LongTensor(range(87, 119))

        self.table = table

    def get_ptable_coords(self, Z):
        row, col = torch.zeros_like(Z), torch.zeros_like(Z)

        for atype in torch.unique(Z):
            if atype == 0:
                continue
            loc = torch.where(atype == self.table)
            if loc[0].size()[0] == loc[1].size()[0] == 0:
                print(f"ATYPE {atype} not in PTABLE")
                row[Z == atype], col[Z == atype] = (0, 0)
            else:
                row[Z == atype], col[Z == atype] = loc

        return row, col

    def forward(self, Z):
        row, col = self.get_ptable_coords(Z)
        row_em = self.row_embedding(row)
        col_em = self.col_embedding(col)
        return 0.5 * (row_em + col_em)


if __name__ == "__main__":
    emb = ptable_embedding()
    Z = torch.repeat_interleave(
        torch.tensor([[x for x in range(1, 119)]], dtype=int), 100, axis=0
    )
    a = emb(Z).detach().numpy()
    print(a)

    plt.scatter(a[0, ::, 0], a[0, ::, 1], c=(Z[0] / 118.0))
    plt.grid()
    plt.colorbar()
    plt.show()
