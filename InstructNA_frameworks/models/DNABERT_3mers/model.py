import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig,PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutput,MaskedLMOutput
from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoTokenizer,AutoModelForMaskedLM

class InstructNAConfig(PretrainedConfig):
    model_type = "instructna_DNABERT_3mers"

    def __init__(
        self,
        proj_dim=8,
        decoder_hidden_dim=768,
        decoder_n_layers=6,
        decoder_n_heads=8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.HF_MODEL_NAME = "zhihan1996/DNA_bert_3"
        self.proj_dim = proj_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_n_layers = decoder_n_layers
        self.decoder_n_heads = decoder_n_heads

        ## jointly training settings
        self.jointly_training = False
        self.mlm_alpha = 1.0  # Weight for MLM loss when jointly training
        
        self.CLS_embd=False # Whether to use CLS token embedding as sequence representation
        self.seq_lengths=20 # if CLS_embd is True, set seq_lengths to the fixed token nums in a sequence
                            # Beacause SELEX sequences are all fixed length, we can directly set it here.
        

        ## two stage training settings
        self.train_encoder = True
        self.train_decoder = False
        

class EncoderWithProj(nn.Module):
    def __init__(
        self,
        HF_MODEL: PreTrainedModel,
        proj_dim: int,
        CLS_embd: bool = False,
    ):
        """
        Encoder with latent projection and optional MLM loss.

        Args:
            hf_model: HuggingFace pretrained MLM model
            tokenizer: Corresponding tokenizer
            lm_hidden_dim: Hidden size of the LM
            proj_dim: Latent projection dimension
        """
        super().__init__()

        print("Loading pretrained model:", HF_MODEL)
        self.lm = AutoModelForMaskedLM.from_pretrained(HF_MODEL, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
        print("Pretrained model loaded.")
        
        self.down_proj = nn.Linear(self.lm.config.hidden_size, proj_dim)
        self.down_lm_head=nn.Linear(proj_dim,self.tokenizer.vocab_size)

        self.CLS_embd = CLS_embd
        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor=None,
        mlm_labels: torch.LongTensor=None ,
    ):
        """
        Args:
            input_ids: (B, L)
            attention_mask: (B, L)
            mlm_labels: (B, L), only masked positions have valid labels

        Returns:
            dict with keys:
                latent: (B, L, proj_dim)
                logits: (B, L, vocab_size)
                mlm_loss: scalar or None
        """

        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Last-layer hidden states → latent space
        hidden_states = outputs.hidden_states[-1].float()
        latent = self.down_proj(hidden_states)
        logits = self.down_lm_head(latent)
        if self.CLS_embd:
            latent = latent[:, 0, :]  
        

        mlm_loss = None
        if mlm_labels is not None:
            mlm_loss = self.mlm_loss_fn(
                logits.view(-1, logits.size(-1)),
                mlm_labels.view(-1),
            )

        return {
            "mlm_output": MaskedLMOutput(
                loss=mlm_loss,
                logits=outputs.logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            ),
            "latent": latent,
        }




class TransformerWithProj(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        n_layers,
        n_heads,
        vocab_size,
        CLS_embd: bool = False,
    ):
        super().__init__()

        self.up_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.lm_head = nn.Linear(hidden_dim, vocab_size)

        self.CLS_embd = CLS_embd
    def forward(self, latent, attention_mask=None, seq_lengths=None):
        
        if self.CLS_embd and seq_lengths is not None:
            latent = latent.view(latent.size(0), 1, latent.size(-1)).repeat(1, seq_lengths, 1)
            
        x = self.up_proj(latent)

        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        x = self.transformer(
            x,
            src_key_padding_mask=key_padding_mask,
        )

        logits = self.lm_head(x)
        return logits



class InstructNA(PreTrainedModel):
    config_class = InstructNAConfig

    def __init__(self, config: InstructNAConfig):
        super().__init__(config)
        self.config = config

        self.encoder = EncoderWithProj(
                HF_MODEL=config.HF_MODEL_NAME,
                proj_dim=config.proj_dim,
                CLS_embd=config.CLS_embd,
        )
        self.tokenizer = self.encoder.tokenizer
        
        self.decoder = TransformerWithProj(
            input_dim=config.proj_dim,
            hidden_dim=config.decoder_hidden_dim,
            n_layers=config.decoder_n_layers,
            n_heads=config.decoder_n_heads,
            vocab_size=self.tokenizer.vocab_size,
            CLS_embd=config.CLS_embd,
        )

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.post_init()  

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None, # labels default to MLM labels
        full_input_ids=None,
    ):
        
        # Encode
        encoder_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mlm_labels=labels,
        )
        
        mlm_loss = encoder_out["mlm_output"].loss
        logits = encoder_out["mlm_output"].logits
        latent = self.encoder(
            input_ids=full_input_ids,
            attention_mask=attention_mask,
            mlm_labels=labels,
        )["latent"]
        
        
        rec_loss = None
        if self.config.jointly_training or self.config.train_decoder:
            # Decode
            logits = self.decoder(
                latent,
                attention_mask=attention_mask,
                seq_lengths=self.config.seq_lengths,
            )

            rec_loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                full_input_ids.view(-1),
            )
            
        loss = None
        
        if self.config.jointly_training:
            loss = rec_loss + self.config.mlm_alpha * mlm_loss
        elif self.config.train_decoder and not self.config.jointly_training:
            loss = rec_loss
        elif self.config.train_encoder and not self.config.jointly_training:
            loss = mlm_loss
            
        output = {
                "loss": loss,
                "logits": logits,
                "mlm_loss": mlm_loss,
            }

        if rec_loss is not None:
            output["rec_loss"] = rec_loss

        return output
    
    def encode_into_latent(self, input_ids, attention_mask=None):
        encoder_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mlm_labels=None,
        )
        latent = encoder_out["latent"]
        return latent
    
    def decode_from_latent(self, latent, attention_mask=None):
        logits = self.decoder(
            latent,
            attention_mask=attention_mask,
            seq_lengths=self.config.seq_lengths if self.config.CLS_embd else None,
        )
        return logits

    @property
    def train_encoder(self):
        return self.config.train_encoder

    @train_encoder.setter
    def train_encoder(self, value):
        self.config.train_encoder = value
        if value:
            self.config.train_decoder = False

    @property
    def train_decoder(self):
        return self.config.train_decoder
    
    @train_decoder.setter
    def train_decoder(self, value):
        self.config.train_decoder = value
        if value:
            self.config.train_encoder = False