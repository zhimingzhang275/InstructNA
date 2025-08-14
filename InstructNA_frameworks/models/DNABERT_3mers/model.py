import torch
import torch.nn as nn
from transformers import AutoModel

class EncoderWithProj(nn.Module):
    def __init__(self, lm, lm_embeds_dim, proj_dim, lm_vocab_size=512,is_evo=None):
        super().__init__()
        self.lm = lm
        self.down_proj = nn.Linear(lm_embeds_dim, proj_dim)

        self.lm_vocab_size=lm_vocab_size
        self.last_head=nn.Linear(proj_dim,lm_vocab_size)

    def forward(self, x, predict_token=None,):

        x = self.lm(x, output_hidden_states=True)['hidden_states'][-1]
        x=x.to(torch.float32)
        x = self.down_proj(x)
        if predict_token:
            x=self.last_head(x)
        return x


class TransformerWithProj(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=768, n_layers=6, n_heads=8,lm_vocab_size=512):
        super().__init__()
        self.up_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.lm_vocab_size=lm_vocab_size
        self.last_head=nn.Linear(hidden_dim,lm_vocab_size)
        
    def forward(self, x,):
        x = self.up_proj(x)
        x = self.transformer(x)
        x=self.last_head(x)
        return x

class InstructNA(nn.Module):
    def __init__(self, lm=None,lm_embeds_dim=512, lm_vocab_size=512,decoder_hidden_dim=768, n_layers=6, n_heads=8,proj_dim=8):
        super(InstructNA, self).__init__()
        self.encoder = EncoderWithProj(lm, lm_embeds_dim, proj_dim=proj_dim, lm_vocab_size=lm_vocab_size)
        
        self.decoder=TransformerWithProj(
                                        input_dim=proj_dim,
                                        hidden_dim=decoder_hidden_dim,
                                        n_layers=n_layers,
                                        n_heads=n_heads,
                                        lm_vocab_size=lm_vocab_size
                                    )
        
        self.loss_fn=nn.CrossEntropyLoss()

    def forward(self, x):
        embeds = self.encoder(x)

        x = self.decoder(embeds)                  

        return x
    

