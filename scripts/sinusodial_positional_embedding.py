# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from traceback import print_tb
from typing import Any, Optional

import torch
import torch.onnx.operators
from torch import Tensor, embedding, nn

from util import make_positions


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024, sinpostype="ratio"):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx
        )
        self.onnx_trace = False
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.max_positions = init_size
        self.sinpostype = sinpostype

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None, length=None,
    ):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        if length is None:
            #default positional encoding
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
            emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
                1
            ) * emb.unsqueeze(0)
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
                num_embeddings, -1
            )
        else:
            #represent length by sinusoidal pos
            emb = length.float().log() / (half_dim - 1) #batch
            emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=emb.device).unsqueeze(0) * -emb.unsqueeze(1)) #batch * dim
            wave = torch.arange(num_embeddings, dtype=torch.float, device=emb.device).unsqueeze(0).expand(emb.size(0), num_embeddings)
            emb = wave.unsqueeze(2) * emb.unsqueeze(1) #batch * len * dim
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=2).view(emb.size(0), num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            if length is None:
                emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
            else:
                emb = torch.cat([emb, torch.zeros(num_embeddings, 1, device=emb.device)], dim=2)
        if padding_idx is not None:
            if length is None:
                emb[padding_idx, :] = 0
            else:
                emb[:, padding_idx, :] = 0
        return emb

    def forward(
        self,
        input_shape,
        incremental_state: Optional[Any] = None,
        length = None,
        timestep: Optional[Tensor] = None,
        positions: Optional[Any] = None,
    ):
        # print("length: ", length)
        # print("Input shape ", input_shape)
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_shape[:2]
        max_pos = self.padding_idx + 1 + seq_len
        if length is not None and self.sinpostype == 'ratio':
            length4getemb = length
        else:
            length4getemb = None
        if self.weights is None or length4getemb is not None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx, length4getemb,
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if length4getemb is None and self.sinpostype == None:
                if self.onnx_trace:
                    return (
                        self.weights.index_select(index=self.padding_idx + pos, dim=0)
                        .unsqueeze(1)
                        .repeat(bsz, 1, 1)
                    )
                return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)
            elif self.sinpostype == 'absolute':
                minuspos = (length.view(-1) + 3) - (self.padding_idx + pos).type_as(length.data)
                return self.weights.index_select(0, minuspos.view(-1)).view(bsz, 1, -1)
            else:
                return self.weights[:, self.padding_idx + pos, :]
        positions = torch.arange(
            seq_len, dtype=torch.long, device=self.weights.device
        )
        if self.sinpostype == 'absolute':
            # when absolute
            minuspos = (length.view(-1)) - (positions).type_as(length.data)
            minuspos[minuspos < 0] = 0
            return self.weights.squeeze().index_select(0, minuspos.view(-1)).view(seq_len, self.embedding_dim).repeat(bsz, 1, 1).detach()



        return self.weights.squeeze().index_select(0, positions).view(seq_len, self.embedding_dim).repeat(bsz, 1, 1).detach()
        
        # positions = make_positions(
        #     input, self.padding_idx, left_pad=False, onnx_trace=self.onnx_trace
        # )
        # if length4getemb is None and sinpostype == None:
        #     if self.onnx_trace:
        #         flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
        #         embedding_shape = torch.cat(
        #             (bsz.view(1), seq_len.view(1), torch.tensor([-1], dtype=torch.long))
        #         )
        #         embeddings = torch.onnx.operators.reshape_from_tensor_shape(
        #             flat_embeddings, embedding_shape
        #         )
        #         return embeddings
        #     return (
        #         self.weights.index_select(0, positions.view(-1))
        #         .view(bsz, seq_len, -1)
        #         .detach()
        #     )
        # elif sinpostype == 'absolute':
        #     #add 3 to set range value with positions (if no value addition, cause error due to index -1)
        #     minuspos = (length.view(-1, 1) + 3).expand(bsz, seq_len) - positions.view(bsz, seq_len)
        #     return self.weights.index_select(0, minuspos.view(-1)).view(bsz, seq_len, -1).detach()
        # else:
        #     return self.weights.index_select(1, positions[0]).view(bsz, seq_len, -1).detach()