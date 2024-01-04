import torch
import torch.nn as nn
from beats.Tokenizer import TokenizersConfig, Tokenizers
from beats.BEATS import BEATs, BEATsConfig

class AudioEncoder(torch.nn.Module):
    def __init__(self,model_single_modality=False):
        super().__init__()

        # Load pretrained BEATs model
        checkpoint = torch.load('beats/BEATs_iter3_plus_AS2M.pt')
        cfg = BEATsConfig(checkpoint['cfg'])
        self.BEATs_model = BEATs(cfg)
        self.BEATs_model.load_state_dict(checkpoint['model'])

        # Update config to use finetuned model
        new_cfg = BEATsConfig(checkpoint['cfg'])
        new_cfg.finetuned_model = True
        new_cfg.predictor_class = 3
        self.BEATs_model.update_cfg(new_cfg)

        n_embeddings = 300
        self.dropout = nn.Dropout(p=0.2)
        self.single_modalitity = model_single_modality
        # A classifier is added to the model
        if model_single_modality:
            self.linear = nn.Linear(768, n_embeddings)
            self.classifier = nn.Linear(n_embeddings, 3)

    def forward(self, prosody,pad):

        # Extract embeddings from BEATs model
        x, padding_mask = self.BEATs_model.extract_features(prosody, padding_mask=pad, return_embeddings=True)

        if self.single_modalitity:
            x[padding_mask.bool()] = 0
            x = x.sum(dim=1)
            x = x / (~padding_mask.bool().sum(dim=1).unsqueeze(-1).expand_as(x))
            x = self.dropout(x)
            x = self.linear(x)
            x = self.dropout(x)
            x = nn.functional.leaky_relu(x)
            x = self.classifier(x)

            return x

        return x, padding_mask

