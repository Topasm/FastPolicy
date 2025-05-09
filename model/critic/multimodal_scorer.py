import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import math
from dataclasses import dataclass
import einops


# --- Configuration ---
@dataclass
class MultimodalScorerConfig:
    state_dim: int
    image_feature_dim: int  # Dimension of pre-extracted image features
    # Max length for state sequence (horizon), image sequence (n_obs_steps), language tokens
    max_state_len: int = 16
    max_image_len: int = 2
    max_lang_len: int = 77
    # Transformer settings (used for state sequence encoding)
    state_encoder_hidden_dim: int = 768  # Dim for state encoder transformer
    state_encoder_num_layers: int = 4   # Fewer layers for state sequence
    state_encoder_num_heads: int = 12
    # Context settings
    context_hidden_dim: int = 768  # Dim for language/image context
    # Final MLP settings
    combined_hidden_dim: int = 1024  # Hidden dim for the final scoring MLP
    swiglu_intermediate_factor: int = 4  # Factor for SwiGLU intermediate dim
    dropout: float = 0.1
    tokenizer_name: str = "gpt2"
    # Allow changing the number of state encoder layers
    # User requested parameter, overrides state_encoder_num_layers if provided
    num_layers: int = 8


# --- SwiGLU Activation ---
class SwiGLU(nn.Module):
    """ SwiGLU Activation Function """

    def __init__(self, dim: int, hidden_dim: int | None = None, bias: bool = False):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else int(
            2/3 * dim * 4)  # Common SwiGLU intermediate size
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# --- Simple Attention Pooling ---
class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = False):
        super().__init__()
        self.query = nn.Parameter(torch.randn(
            1, 1, output_dim))  # Learnable query
        self.key_proj = nn.Linear(input_dim, output_dim, bias=bias)
        self.value_proj = nn.Linear(input_dim, output_dim, bias=bias)
        self.scale = output_dim ** -0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, Seq, Dim)
            mask: Attention mask (B, Seq), True where tokens should be ignored.
        Returns:
            Pooled output (B, Dim_out)
        """
        B, _, _ = x.shape
        k = self.key_proj(x)    # (B, Seq, Dim_out)
        v = self.value_proj(x)  # (B, Seq, Dim_out)
        q = self.query.expand(B, -1, -1)  # (B, 1, Dim_out)

        # Attention scores (B, 1, Seq)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # Mask expects True for positions NOT to attend to.
            # Add mask dimension for query head.
            attn_scores = attn_scores.masked_fill(
                mask.unsqueeze(1), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, 1, Seq)

        # Weighted sum of values (B, 1, Dim_out)
        pooled_output = attn_weights @ v
        return pooled_output.squeeze(1)  # (B, Dim_out)


# --- Main Model ---
class MultimodalTrajectoryScorer(nn.Module):
    def __init__(self, config: MultimodalScorerConfig):
        super().__init__()
        self.config = config

        # Override state encoder layers if num_layers is explicitly set differently
        state_encoder_layers = config.num_layers if config.num_layers != 8 else config.state_encoder_num_layers
        state_encoder_dim = config.state_encoder_hidden_dim
        context_dim = config.context_hidden_dim
        use_bias = False  # ModernBERT: No bias in linear layers

        # 1. Tokenizer & Language Embedding/Projection
        self.tokenizer = tiktoken.get_encoding(config.tokenizer_name)
        vocab_size = self.tokenizer.n_vocab
        lang_embed_dim = 512
        self.lang_embed = nn.Embedding(vocab_size, lang_embed_dim)
        # Use Attention Pooling instead of projection after mean pooling
        self.lang_pool = AttentionPooling(
            lang_embed_dim, context_dim, bias=use_bias)
        self.lang_pos_embed = nn.Parameter(torch.randn(
            1, config.max_lang_len, lang_embed_dim) * 0.02)

        # 2. Image Projection
        self.image_proj = nn.Linear(
            config.image_feature_dim, context_dim, bias=use_bias)

        # 3. State Sequence Encoder (Transformer)
        self.state_proj = nn.Linear(
            config.state_dim, state_encoder_dim, bias=use_bias)
        self.state_pos_embed = nn.Parameter(torch.randn(
            1, config.max_state_len, state_encoder_dim) * 0.02)
        self.state_cls_token = nn.Parameter(
            torch.randn(1, 1, state_encoder_dim))

        state_encoder_layer = nn.TransformerEncoderLayer(
            d_model=state_encoder_dim,
            nhead=config.state_encoder_num_heads,
            dim_feedforward=state_encoder_dim * 4,
            dropout=config.dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True
        )
        self.state_transformer_encoder = nn.TransformerEncoder(
            state_encoder_layer,
            num_layers=state_encoder_layers,
            norm=nn.LayerNorm(state_encoder_dim)
        )

        # 4. Combined MLP Head using SwiGLU
        combined_input_dim = context_dim + state_encoder_dim
        mlp_hidden_dim = config.combined_hidden_dim
        swiglu_intermediate_dim = int(
            mlp_hidden_dim * config.swiglu_intermediate_factor * (2/3))

        self.output_head = nn.Sequential(
            nn.LayerNorm(combined_input_dim),
            nn.Linear(combined_input_dim,
                      swiglu_intermediate_dim * 2, bias=use_bias),
            nn.SiLU(),
            # Fixed dimension mismatch
            nn.Linear(swiglu_intermediate_dim * 2, 1, bias=use_bias)
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize projections and output head
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        # Positional embeddings and CLS token are already initialized via nn.Parameter

    def encode_context(
        self,
        image_features: torch.Tensor,      # (B, T_image, D_image)
        lang_instruction: list[str],       # List of strings [B]
        # (B, T_image), True where padded
        image_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """ Encodes language and image history into a context vector. """
        B = image_features.shape[0]
        device = image_features.device

        # --- Encode Language ---
        # Replace empty or whitespace-only strings with EOT token string
        processed_lang_instruction = []
        for text in lang_instruction:
            if not text or text.isspace():
                processed_lang_instruction.append(
                    self.tokenizer.decode([self.tokenizer.eot_token]))
            else:
                processed_lang_instruction.append(text)

        lang_tokens = [self.tokenizer.encode(
            text, allowed_special="all") for text in processed_lang_instruction]

        # Ensure no empty token lists after encoding (should be handled by placeholder)
        for i, tokens in enumerate(lang_tokens):
            if not tokens:
                lang_tokens[i] = [self.tokenizer.eot_token]

        max_len = min(max(len(tokens)
                      for tokens in lang_tokens), self.config.max_lang_len)
        padded_tokens = torch.full(
            (B, max_len), self.tokenizer.eot_token, dtype=torch.long, device=device)
        lang_attn_mask = torch.zeros(
            (B, max_len), dtype=torch.bool, device=device)
        for i, tokens in enumerate(lang_tokens):
            seq = tokens[:max_len]
            if seq:  # Ensure seq is not empty before tensor conversion
                padded_tokens[i, :len(seq)] = torch.tensor(
                    seq, dtype=torch.long, device=device)
            lang_attn_mask[i, len(seq):] = True

        lang_emb = self.lang_embed(padded_tokens)  # (B, T_lang, D_lang_embed)

        lang_emb = lang_emb + self.lang_pos_embed[:, :max_len, :]

        # Use Attention Pooling for language context
        lang_context = self.lang_pool(
            lang_emb, mask=lang_attn_mask)  # (B, D_context)

        # --- Encode Image ---
        image_emb = self.image_proj(image_features)  # (B, T_image, D_context)
        if image_padding_mask is not None:
            image_emb = image_emb.masked_fill(
                image_padding_mask.unsqueeze(-1), 0.0)
            num_unmasked = (~image_padding_mask).sum(dim=1, keepdim=True)
            image_context = image_emb.sum(
                dim=1) / num_unmasked.clamp(min=1)  # Added clamp for safety
        else:
            image_context = image_emb.mean(dim=1)

        # --- Combine Context ---
        context = lang_context + image_context  # (B, D_context)

        return context

    def encode_state_sequence(
        self,
        state_sequence: torch.Tensor,      # (B * N_seq, T_state, D_state)
        # (B * N_seq, T_state), True where padded
        state_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """ Encodes a batch of state sequences using a Transformer. """
        BN, T_state, _ = state_sequence.shape
        device = state_sequence.device

        state_emb = self.state_proj(state_sequence)

        state_emb = state_emb + self.state_pos_embed[:, :T_state, :]

        cls_token = self.state_cls_token.expand(BN, -1, -1)
        state_emb_with_cls = torch.cat([cls_token, state_emb], dim=1)

        if state_padding_mask is None:
            state_padding_mask = torch.zeros(
                (BN, T_state), dtype=torch.bool, device=device)

        cls_mask = torch.zeros((BN, 1), dtype=torch.bool, device=device)
        combined_attn_mask = torch.cat([cls_mask, state_padding_mask], dim=1)

        transformer_output = self.state_transformer_encoder(
            state_emb_with_cls,
            src_key_padding_mask=combined_attn_mask
        )

        state_encoding = transformer_output[:, 0, :]  # CLS token output

        return state_encoding

    def forward(
        self,
        state_sequences: torch.Tensor,     # (B, N_seq, T_state, D_state)
        image_features: torch.Tensor,      # (B, T_image, D_image) - History
        lang_instruction: list[str],       # List of strings [B]
        state_padding_mask: torch.Tensor | None = None,
        image_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Scores multiple state sequences for each language/image context.

        Returns:
            Tensor of shape (B, N_seq) with scores.
        """
        B, N_seq, T_state, _ = state_sequences.shape

        context_encoding = self.encode_context(
            image_features, lang_instruction, image_padding_mask)

        state_sequences_flat = einops.rearrange(
            state_sequences, 'b n t d -> (b n) t d')
        state_padding_mask_flat = None
        if state_padding_mask is not None:
            state_padding_mask_flat = einops.rearrange(
                state_padding_mask, 'b n t -> (b n) t')

        state_encoding_flat = self.encode_state_sequence(
            state_sequences_flat, state_padding_mask_flat)

        context_encoding_repeated = einops.repeat(
            context_encoding, 'b d -> b n d', n=N_seq)
        state_encoding = einops.rearrange(
            state_encoding_flat, '(b n) d -> b n d', n=N_seq)

        combined_features = torch.cat(
            [context_encoding_repeated, state_encoding], dim=-1)

        combined_features_flat = einops.rearrange(
            combined_features, 'b n d -> (b n) d')
        scores_flat = self.output_head(combined_features_flat)

        scores = einops.rearrange(scores_flat, '(b n) 1 -> b n', n=N_seq)

        return scores

    @torch.no_grad()
    def score(
        self,
        state_sequences: torch.Tensor,     # (B, N_seq, T_state, D_state)
        image_features: torch.Tensor,      # (B, T_image, D_image) - History
        lang_instruction: list[str],       # List of strings [B]
        state_padding_mask: torch.Tensor | None = None,
        image_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """ Inference entry point. Returns scores of shape (B, N_seq). """
        self.eval()
        scores = self.forward(
            state_sequences,
            image_features,
            lang_instruction,
            state_padding_mask,
            image_padding_mask
        )
        return scores
