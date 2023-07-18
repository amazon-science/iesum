# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl


class GAT(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GAT, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g):
        z = self.fc(g.ndata['hn'])
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return g


if __name__ == "__main__":
    u1, v1 = torch.tensor([0, 0, 0, 1, 1]), torch.tensor([1, 2, 3, 3, 2])
    g1 = dgl.graph((u1, v1))
    g1.ndata["hn"] = torch.rand(4,6)
    g1.edata["he"] = torch.rand(5,8)
    print(g1.ndata)
    print(g1.edata)

    u2, v2 = torch.tensor([0, 1, 0, 1, 1, 3, 2, 4]), torch.tensor([1, 0, 3, 3, 2, 4, 4, 1])
    g2 = dgl.graph((u2, v2))
    g2.ndata["hn"] = torch.rand(5,6)
    g2.edata["he"] = torch.rand(8,8)
    print(g2.ndata)
    print(g2.edata)

    u3, v3 = torch.tensor([0, 1, 0]), torch.tensor([1, 0, 2])
    g3 = dgl.graph((u3, v3))
    g3.ndata["hn"] = torch.rand(3,6)
    g3.edata["he"] = torch.rand(3,8)
    print(g3.ndata)
    print(g3.edata)

    gb = dgl.batch([g1, g2, g3])

    print(gb.num_nodes())
    print(gb.num_edges())
    print(gb.ndata)
    print(gb.edata)

    gs = dgl.unbatch(gb)
    print(gs)
    print(gs[1].ndata)
    print(gs[1].edata)


