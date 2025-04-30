import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class CriticBERT(nn.Module):
    # BERT-based critic model that scores trajectories conditioned on language instructions.

    def __init__(self, state_dim: int, horizon: int, hidden_dim: int = 768, bert_model_name: str = 'bert-base-uncased'):
        super().__init__()
        self.state_dim = state_dim
        self.horizon = horizon
        self.trajectory_input_dim = state_dim * horizon
        self.hidden_dim = hidden_dim

        # BERT model for language processing
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        # Freeze BERT parameters if desired
        # for param in self.bert_model.parameters():
        #     param.requires_grad = False
        bert_output_dim = self.bert_model.config.hidden_size

        # MLP for processing flattened trajectory
        self.trajectory_mlp = nn.Sequential(
            nn.Linear(self.trajectory_input_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Combined MLP to process trajectory and language embeddings
        self.combined_mlp = nn.Sequential(
            nn.Linear(hidden_dim // 2 + bert_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, trajectories: torch.Tensor, lang_instructions: list[str]) -> torch.Tensor:
        # Forward pass for the CriticBERT model.
        # Args:
        #     trajectories: Tensor of shape[N, horizon, state_dim]
        #     lang_instructions: List of N language instruction strings.
        # Returns:
        #     Tensor of shape[N] representing the scores for each trajectory.
        N = trajectories.shape[0]
        device = trajectories.device

        # 1. Process trajectories
        flat_trajectories = trajectories.view(N, -1)
        trajectory_embedding = self.trajectory_mlp(
            flat_trajectories)  # [N, hidden_dim // 2]

        # 2. Process language instructions
        # Ensure tokenizer runs on the correct device if possible, or move inputs
        inputs = self.bert_tokenizer(
            lang_instructions,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512  # Standard max length for BERT
        ).to(device)

        with torch.no_grad():  # Or fine-tune BERT by removing no_grad()
            outputs = self.bert_model(**inputs)
        # Use the [CLS] token embedding as the language representation
        # [N, bert_output_dim]
        lang_embedding = outputs.last_hidden_state[:, 0, :]

        # 3. Combine embeddings and score
        combined_embedding = torch.cat(
            [trajectory_embedding, lang_embedding], dim=1)
        scores = self.combined_mlp(combined_embedding).squeeze(-1)  # [N]

        return scores


class CriticBERTScorer:
    # Loads a trained CriticBERT model and scores candidate trajectories based on language.

    def __init__(
        self,
        model_path: str,
        state_dim: int,
        horizon: int,
        hidden_dim: int = 768,
        bert_model_name: str = 'bert-base-uncased',
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = CriticBERT(
            state_dim=state_dim,
            horizon=horizon,
            hidden_dim=hidden_dim,
            bert_model_name=bert_model_name
        ).to(self.device)
        self.model.load_state_dict(torch.load(
            model_path, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def score(self, trajectories: torch.Tensor, lang_instructions: list[str]) -> torch.Tensor:
        # Scores trajectories based on language instructions.
        # Args:
        #     trajectories: Tensor of shape[N, horizon, state_dim]
        #     lang_instructions: List of N language instruction strings.
        # Returns:
        #     Tensor of shape[N] representing the critic scores.
        # Ensure trajectories are on the correct device
        trajectories = trajectories.to(self.device)

        # The model's forward pass handles tokenization and BERT processing
        vals = self.model(trajectories, lang_instructions)
        return vals
