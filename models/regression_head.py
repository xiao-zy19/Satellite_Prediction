"""
Regression head for extracting numerical predictions from MLLM outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import re


class RegressionHead(nn.Module):
    """
    Regression head that can be added to MLLM for direct numerical prediction.

    Can be used as an alternative to parsing text output.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 1,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        """
        Initialize Regression Head.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Number of outputs (1 for single regression)
            dropout: Dropout probability
            num_layers: Number of MLP layers
        """
        super().__init__()

        layers = []
        in_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features, shape (B, input_dim) or (B, seq_len, input_dim)

        Returns:
            Regression output, shape (B, output_dim)
        """
        if x.dim() == 3:
            # Pool over sequence dimension
            x = x.mean(dim=1)

        return self.mlp(x)


class TextToRegressionParser:
    """
    Parser to extract numerical predictions from text output.
    """

    def __init__(
        self,
        patterns: Optional[list] = None,
        default_value: float = 0.0,
    ):
        """
        Initialize parser.

        Args:
            patterns: List of regex patterns to match numbers
            default_value: Value to return if no number found
        """
        self.default_value = default_value

        # Default patterns for population growth rate
        self.patterns = patterns or [
            r"预测[值为]*[：:]\s*([+-]?\d+\.?\d*)[\s‰%]*",
            r"([+-]?\d+\.?\d*)[\s]*[‰%]",
            r"增长率[为是]*[：:]\s*([+-]?\d+\.?\d*)",
            r"[-+]?\d+\.?\d*",  # Fallback: any number
        ]

    def parse(self, text: str) -> float:
        """
        Parse numerical value from text.

        Args:
            text: Model output text

        Returns:
            Extracted numerical value
        """
        for pattern in self.patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1) if match.lastindex else match.group())
                except (ValueError, IndexError):
                    continue

        return self.default_value

    def parse_batch(self, texts: list) -> torch.Tensor:
        """
        Parse a batch of texts.

        Args:
            texts: List of text strings

        Returns:
            Tensor of parsed values, shape (B,)
        """
        values = [self.parse(t) for t in texts]
        return torch.tensor(values)


class HybridRegressionModel(nn.Module):
    """
    Hybrid model that combines MLLM with a regression head.

    Can output both text and direct numerical predictions.
    """

    def __init__(
        self,
        mllm_model: nn.Module,
        regression_input_dim: int = 1024,
        use_pooled_output: bool = True,
    ):
        """
        Initialize Hybrid model.

        Args:
            mllm_model: Base MLLM model
            regression_input_dim: Dimension for regression head
            use_pooled_output: Whether to use pooled output or last hidden state
        """
        super().__init__()

        self.mllm = mllm_model
        self.use_pooled_output = use_pooled_output

        # Regression head
        self.regression_head = RegressionHead(
            input_dim=regression_input_dim,
            hidden_dim=256,
            output_dim=1,
        )

        # Parser for text output
        self.parser = TextToRegressionParser()

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_text: bool = False,
        **kwargs
    ) -> dict:
        """
        Forward pass.

        Args:
            pixel_values: Image tensor
            input_ids: Input token IDs
            attention_mask: Attention mask
            return_text: Whether to also return text output
            **kwargs: Additional arguments

        Returns:
            Dictionary with 'regression' and optionally 'text' outputs
        """
        outputs = {}

        # Get MLLM hidden states
        mllm_outputs = self.mllm(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )

        # Get features for regression
        if self.use_pooled_output and hasattr(mllm_outputs, 'pooler_output'):
            features = mllm_outputs.pooler_output
        else:
            # Use mean of last hidden state
            hidden_states = mllm_outputs.hidden_states[-1]
            features = hidden_states.mean(dim=1)

        # Regression prediction
        outputs['regression'] = self.regression_head(features)

        # Text output if needed
        if return_text:
            outputs['logits'] = mllm_outputs.logits

        return outputs

    def compute_loss(
        self,
        outputs: dict,
        labels: torch.Tensor,
        text_labels: Optional[torch.Tensor] = None,
        regression_weight: float = 0.5,
    ) -> dict:
        """
        Compute combined loss.

        Args:
            outputs: Model outputs
            labels: Regression labels
            text_labels: Optional text labels for language modeling loss
            regression_weight: Weight for regression loss

        Returns:
            Dictionary with losses
        """
        losses = {}

        # Regression loss
        reg_pred = outputs['regression'].squeeze()
        reg_loss = F.mse_loss(reg_pred, labels)
        losses['regression_loss'] = reg_loss

        # Language modeling loss if applicable
        if 'logits' in outputs and text_labels is not None:
            lm_loss = F.cross_entropy(
                outputs['logits'].view(-1, outputs['logits'].size(-1)),
                text_labels.view(-1),
                ignore_index=-100
            )
            losses['lm_loss'] = lm_loss

            # Combined loss
            losses['total_loss'] = (
                regression_weight * reg_loss +
                (1 - regression_weight) * lm_loss
            )
        else:
            losses['total_loss'] = reg_loss

        return losses


class ConsistencyLoss(nn.Module):
    """
    Loss to ensure consistency between text output and regression head output.
    """

    def __init__(self, parser: Optional[TextToRegressionParser] = None):
        super().__init__()
        self.parser = parser or TextToRegressionParser()

    def forward(
        self,
        regression_output: torch.Tensor,
        text_outputs: list,
        weight: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute consistency loss between regression and parsed text values.

        Args:
            regression_output: Direct regression predictions
            text_outputs: List of generated text outputs
            weight: Loss weight

        Returns:
            Consistency loss
        """
        # Parse values from text
        text_values = self.parser.parse_batch(text_outputs)
        text_values = text_values.to(regression_output.device)

        # MSE between regression output and parsed values
        loss = F.mse_loss(regression_output.squeeze(), text_values)

        return weight * loss
