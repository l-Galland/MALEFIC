from transformers import RobertaModel, BertModel
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
F=16
F2 = 16
n_conv=5
import math
class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class AUEncoder(torch.nn.Module):
    def __init__(self,model_single_modality=False):

        super().__init__()

        self.conv0 = nn.Conv2d(16,F,(1,3), padding=(0,1))
        self.conv1 = nn.Conv2d(F,F,(1,3), padding=(0,1))

        self.d_model= F*16
        self.quantization_size = 10
        self.vocabsize = self.quantization_size *self.d_model

        encoder_layer1 = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=2)
        self.transformer = nn.TransformerEncoder(encoder_layer1, num_layers=1)

        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model,vocab_size=self.vocabsize

        )

    def forward(self, action_units,padding_mask=None):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        action_units = torch.nan_to_num(action_units, nan=0.0, )

        action_units = action_units.unsqueeze(dim=1)
        action_units=Variable(torch.cat(([action_units for i in range(16)]),dim=1),requires_grad=True)

        out = self.conv0(action_units)

        out = nn.functional.leaky_relu(out)
        out = self.conv1(out)

        out = out.permute((3, 0, 1,2))

        out= out.reshape(out.size(0),out.size(1),-1)

        out = torch.sigmoid(out)
        out = self.quantisize(out)
        out = self.pos_encoder(out)

        au_emb = self.transformer(out, src_key_padding_mask =padding_mask)
        x = au_emb.permute((1,0,2))

        return x


    def quantisize(self,x):
        return (x*self.quantization_size).to(int)

