import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from dataclasses import dataclass


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
    state_encoder_num_layers: int = 4   # Base number of layers for state sequence
    state_encoder_num_heads: int = 12
    # Context settings
    context_hidden_dim: int = 768  # Dim for language/image context
    # Final MLP settings
    combined_hidden_dim: int = 1024  # Hidden dim for the final scoring MLP
    # Factor for SwiGLU intermediate dim (multiplier for combined_hidden_dim)
    swiglu_intermediate_factor: int = 4
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
        if hidden_dim is None:
            # Default expansion factor for SwiGLU if not provided (e.g., 4 * 2/3)
            hidden_dim = int(dim * 4 * 2 / 3)
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
            # mask is (B, Seq), True where tokens should be ignored.
            # attn_scores is (B, 1, Seq). We need to apply mask before softmax.
            # Add a large negative number to masked positions.
            attn_scores = attn_scores.masked_fill(
                mask.unsqueeze(1), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, 1, Seq)

        # Weighted sum of values (B, 1, Dim_out)
        pooled_output = attn_weights @ v
        return pooled_output.squeeze(1)  # (B, Dim_out)


# --- Main Model ---
class MultimodalTrajectoryScorer(nn.Module):
    def __init__(self, config: MultimodalScorerConfig = None, state_dim: int = None, horizon: int = None, hidden_dim: int = None):
        super().__init__()

        # Support both config-based and parameter-based initialization
        if config is not None:
            self.config = config
        else:
            # Create a default config from the parameters
            if state_dim is None or horizon is None:
                raise ValueError(
                    "When config is not provided, state_dim and horizon must be specified")

            # Set reasonable defaults
            self.config = MultimodalScorerConfig(
                state_dim=state_dim,
                image_feature_dim=512,  # Default value
                max_state_len=horizon,
                state_encoder_hidden_dim=hidden_dim if hidden_dim is not None else 768,
                context_hidden_dim=hidden_dim if hidden_dim is not None else 768,
                combined_hidden_dim=hidden_dim if hidden_dim is not None else 1024
            )

        # Use num_layers as the definitive setting
        state_encoder_layers = config.num_layers
        state_encoder_dim = config.state_encoder_hidden_dim
        context_dim = config.context_hidden_dim
        use_bias = False  # ModernBERT: No bias in linear layers

        # 1. Tokenizer & Language Embedding/Projection
        self.tokenizer = tiktoken.get_encoding(config.tokenizer_name)
        vocab_size = self.tokenizer.n_vocab
        lang_embed_dim = config.context_hidden_dim  # Align with context_dim
        self.lang_embed = nn.Embedding(vocab_size, lang_embed_dim)
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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=state_encoder_dim,
            nhead=config.state_encoder_num_heads,
            dim_feedforward=state_encoder_dim * 4,  # Standard feedforward expansion
            dropout=config.dropout,
            activation=F.gelu,  # GELU is common in Transformers
            batch_first=True,
            norm_first=True  # Pre-LN
        )
        self.state_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=state_encoder_layers
        )

        # 4. Final MLP
        combined_features_dim = state_encoder_dim + context_dim
        # Calculate SwiGLU hidden dimension using the factor from config
        swiglu_hidden_dim = config.combined_hidden_dim * config.swiglu_intermediate_factor

        self.final_mlp = nn.Sequential(
            nn.Linear(combined_features_dim,
                      config.combined_hidden_dim, bias=use_bias),
            nn.LayerNorm(config.combined_hidden_dim),
            SwiGLU(config.combined_hidden_dim,
                   hidden_dim=swiglu_hidden_dim, bias=use_bias),
            nn.Dropout(config.dropout),
            nn.Linear(config.combined_hidden_dim, 1,
                      bias=use_bias)  # Output a single score
        )
        self.apply(self._init_weights_custom)

    def _init_weights_custom(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine:
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    def encode_context(
        self,
        image_features: torch.Tensor,        # (B, N_img, image_feature_dim)
        lang_instruction: list[str],         # List of B strings
        # (B, N_img) True if padded
        image_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:  # (B, context_hidden_dim)
        device = image_features.device

        # Language encoding
        processed_tokens_list = []
        # Using eot_token for padding. A dedicated pad token ID would be ideal if available and handled by embedding.
        pad_token_id = self.tokenizer.eot_token

        for text in lang_instruction:
            tokens = self.tokenizer.encode(text)
            # Truncate if longer than max_lang_len
            tokens = tokens[:self.config.max_lang_len]
            padding_len = self.config.max_lang_len - len(tokens)
            tokens += [pad_token_id] * padding_len
            processed_tokens_list.append(tokens)

        lang_tokens_tensor = torch.tensor(
            processed_tokens_list, dtype=torch.long, device=device)
        # Create lang_padding_mask: True for pad tokens (to be ignored by AttentionPooling)
        lang_padding_mask = (lang_tokens_tensor ==
                             pad_token_id)  # (B, max_lang_len)

        # (B, max_lang_len, lang_embed_dim)
        lang_embeds = self.lang_embed(lang_tokens_tensor)
        # Add positional embeddings up to the actual sequence length used
        lang_embeds = lang_embeds + \
            self.lang_pos_embed[:, :self.config.max_lang_len, :]
        lang_features = self.lang_pool(
            lang_embeds, mask=lang_padding_mask)  # (B, context_hidden_dim)

        # Image encoding
        # (B, N_img, context_hidden_dim)
        projected_images = self.image_proj(image_features)

        if image_padding_mask is not None:
            # Mask out padded images before averaging
            # image_padding_mask is (B, N_img), True for padded. Unsqueeze for broadcasting.
            projected_images = projected_images.masked_fill(
                image_padding_mask.unsqueeze(-1), 0.0)
            # Count non-padded images for averaging, clamp to avoid division by zero.
            num_valid_images = (~image_padding_mask).sum(
                dim=1, keepdim=True).clamp(min=1)  # (B, 1)
            image_features_pooled = projected_images.sum(
                dim=1) / num_valid_images  # (B, context_hidden_dim)
        else:
            # If no mask, assume all images are valid and average
            image_features_pooled = projected_images.mean(
                dim=1)  # (B, context_hidden_dim)

        # Combine language and image features (e.g., by averaging)
        context_embedding = (lang_features + image_features_pooled) * 0.5
        return context_embedding

    def encode_state_sequence(
        self,
        state_sequence: torch.Tensor,  # (B, Seq_len, state_dim)
        # (B, Seq_len) True where padded
        state_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:  # (B, state_encoder_dim) CLS token output
        B, S, _ = state_sequence.shape
        device = state_sequence.device
        projected_states = self.state_proj(
            state_sequence.float())  # (B, S, state_encoder_dim)

        # Add positional embeddings up to the actual sequence length
        projected_states = projected_states + self.state_pos_embed[:, :S, :]

        cls_tokens = self.state_cls_token.expand(
            B, -1, -1)  # (B, 1, state_encoder_dim)
        # (B, 1+S, state_encoder_dim)
        encoder_input = torch.cat([cls_tokens, projected_states], dim=1)

        # Create padding mask for the transformer
        # True means the position is masked (ignored by attention)
        full_padding_mask = None
        if state_padding_mask is not None:
            # CLS token is never padded
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            full_padding_mask = torch.cat(
                [cls_mask, state_padding_mask], dim=1)  # (B, 1+S)

        # Transformer expects src_key_padding_mask where True indicates a padded token
        transformer_output = self.state_encoder(
            encoder_input, src_key_padding_mask=full_padding_mask)
        state_cls_output = transformer_output[:, 0]  # (B, state_encoder_dim)
        return state_cls_output

    def forward(
        self,
        state_sequences: torch.Tensor,        # (B, Seq_len, state_dim)
        image_features: torch.Tensor,        # (B, N_img, image_feature_dim)
        lang_instruction: list[str],         # List of B strings
        state_padding_mask: torch.Tensor | None = None,  # (B, Seq_len)
        image_padding_mask: torch.Tensor | None = None,  # (B, N_img)
    ) -> tuple[torch.Tensor, torch.Tensor]:  # Returns (scores (B,1), state_cls_embedding (B, state_encoder_dim))

        state_cls_embedding = self.encode_state_sequence(
            state_sequences, state_padding_mask
        )  # (B, state_encoder_dim)

        context_embedding = self.encode_context(
            image_features, lang_instruction, image_padding_mask
        )  # (B, context_dim)

        combined_features = torch.cat(
            [state_cls_embedding, context_embedding], dim=-1)
        scores = self.final_mlp(combined_features)  # (B, 1)

        return scores, state_cls_embedding

    @torch.no_grad()
    def get_raw_state_cls_features(
        self,
        state_sequences: torch.Tensor,
        state_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns the raw CLS token features from the state encoder.
        Sets model to eval mode.
        """
        self.eval()
        state_cls_embedding = self.encode_state_sequence(
            state_sequences, state_padding_mask
        )
        return state_cls_embedding

    @torch.no_grad()
    def score_trajectory(
        self,
        current_state: torch.Tensor,
        action_sequence: torch.Tensor,
        image_features: torch.Tensor = None,
        lang_instruction: list[str] = None,
        state_padding_mask: torch.Tensor | None = None,
        image_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Inference entrypoint for getting scores.
        Sets model to eval mode.

        This method supports two calling patterns:
        1. Full input with state sequences, image features, and language instructions
        2. Simplified input with just current state and action sequence (for action selection)

        Args:
            current_state: Current state tensor, shape [B, state_dim]
            action_sequence: Action sequence tensor, shape [B, horizon, action_dim]
            image_features: Optional image features, shape [B, N_img, image_feature_dim]
            lang_instruction: Optional list of B instruction strings
            state_padding_mask: Optional state padding mask
            image_padding_mask: Optional image padding mask

        Returns:
            scores: Score tensor, shape [B, 1]
        """
        self.eval()

        # Handle the simplified case for action selection
        if image_features is None or lang_instruction is None:
            batch_size = current_state.shape[0]
            device = current_state.device

            # Create placeholder image features (zeros)
            if image_features is None:
                image_features = torch.zeros(
                    batch_size, 1, self.config.image_feature_dim, device=device)
                image_padding_mask = None

            # Create placeholder language instructions (empty strings)
            if lang_instruction is None:
                lang_instruction = [""] * batch_size

            # Prepare state sequence by combining current state with action sequence
            # This is a simplified approach - in a real system you might want to simulate
            # the states resulting from the actions
            horizon = action_sequence.shape[1]

            # Just use current state as the state sequence for scoring
            # Repeat it to fill the sequence
            repeated_state = current_state.unsqueeze(1).repeat(1, horizon, 1)

            # Combine state with action to get a complete representation
            state_sequences = torch.cat(
                [repeated_state, action_sequence], dim=-1)

        # Forward pass to get scores
        scores, _ = self.forward(
            state_sequences,
            image_features,
            lang_instruction,
            state_padding_mask,
            image_padding_mask
        )
        return scores
