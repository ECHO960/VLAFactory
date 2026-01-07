"""
VLA Model wrapper class.

This class wraps LlamaFactory's model architecture and extends it for
Vision-Language-Action tasks.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn


class VLAModel(nn.Module):
    """
    Vision-Language-Action Model wrapper.
    
    This class extends language models from LlamaFactory to support
    vision-language-action learning by adding vision encoder and action head.
    
    Args:
        language_model: Base language model from LlamaFactory
        vision_encoder: Vision encoder for processing images
        action_dim: Dimension of action space
        hidden_dim: Hidden dimension for projection layers
    """

    def __init__(
        self,
        language_model: Optional[nn.Module] = None,
        vision_encoder: Optional[nn.Module] = None,
        action_dim: int = 7,
        hidden_dim: int = 768,
    ):
        super().__init__()
        
        self.language_model = language_model
        self.vision_encoder = vision_encoder
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Vision-language projection layer
        if vision_encoder is not None:
            self.vision_proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.vision_proj = None

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )

        self.value_tokenizer = _ValueTokenizer()
        self.mantissa_embed = nn.Embedding(self.value_tokenizer.mantissa_vocab_size, hidden_dim)
        self.exponent_embed = nn.Embedding(self.value_tokenizer.exponent_vocab_size, hidden_dim)

    def _encode_values(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        token_ids = self.value_tokenizer.encode(values)
        mantissa_ids = token_ids[..., 0]
        exponent_ids = token_ids[..., 1]
        mantissa_embeds = self.mantissa_embed(mantissa_ids)
        exponent_embeds = self.exponent_embed(exponent_ids)
        embeds = torch.stack([mantissa_embeds, exponent_embeds], dim=-2)
        embeds = embeds.reshape(values.shape[0], values.shape[1], -1, embeds.size(-1))
        token_ids = token_ids.reshape(values.shape[0], values.shape[1], -1)
        return token_ids, embeds

    def _build_features_from_batch(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if "state" not in batch or "actions" not in batch:
            raise ValueError("batch must include 'state' and 'actions'.")

        state_tokens, state_embeds = self._encode_values(batch["state"])
        action_tokens, action_embeds = self._encode_values(batch["actions"])

        image_embeds = None
        if "image_embeds" in batch:
            image_embeds = batch["image_embeds"]
        elif "image" in batch and self.vision_encoder is not None:
            image_data = batch["image"]
            if isinstance(image_data, dict):
                views = [image_data[key] for key in sorted(image_data.keys())]
                images = torch.stack(views, dim=2)
            else:
                images = image_data

            if images.dim() != 6:
                raise ValueError("image tensor must be [B, T, V, H, W, C].")
            bsz, time_steps, views, height, width, channels = images.shape
            images = images.reshape(bsz * time_steps * views, height, width, channels)
            if channels == 3:
                images = images.permute(0, 3, 1, 2)
            vision_out = self.vision_encoder(images)
            if vision_out.dim() == 2:
                vision_out = vision_out[:, None, :]
            if vision_out.dim() != 3:
                raise ValueError("vision_encoder must return [B, tokens, C] or [B, C].")
            vision_out = vision_out.reshape(bsz, time_steps, views * vision_out.size(1), vision_out.size(2))
            image_embeds = vision_out

        tokens = [state_tokens, action_tokens]
        embeds = [state_embeds, action_embeds]
        if image_embeds is not None:
            embeds.insert(0, image_embeds)
            image_labels = torch.full(
                image_embeds.shape[:3],
                -100,
                device=image_embeds.device,
                dtype=state_tokens.dtype,
            )
            tokens.insert(0, image_labels)

        features = torch.cat(embeds, dim=2)
        labels = torch.cat(tokens, dim=2)

        instruction_embeds = None
        if "tokenized_prompt" in batch:
            if self.language_model is None:
                raise ValueError("language_model is required to embed tokenized_prompt.")
            token_ids = batch["tokenized_prompt"]
            instruction_embeds = self.language_model.get_input_embeddings()(token_ids)

        attention_mask = batch.get("tokenized_prompt_mask")
        return features, labels, instruction_embeds, attention_mask
        
    def forward(
        self,
        batch: Optional[Dict[str, Any]] = None,
        features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        instruction_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        action_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for VLA model.

        Expected VLA inputs:
            features: [B, T, N, C] bag-of-state features
            labels: [B, T, N] next-step token labels
            instruction_embeds: optional [B, L, C] to prepend as system prompt
        """
        outputs: Dict[str, torch.Tensor] = {}

        if batch is not None:
            if features is not None:
                raise ValueError("Provide either batch or features, not both.")
            features, labels, instruction_embeds, attention_mask = self._build_features_from_batch(batch)

        if features is None:
            # Backward-compatible path for token-based inputs.
            if self.language_model is None:
                raise ValueError("language_model is required when features is None.")
            lm_outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
            if isinstance(lm_outputs, dict):
                outputs.update(lm_outputs)
            else:
                outputs["loss"] = lm_outputs[0] if labels is not None else None
                outputs["logits"] = lm_outputs[1] if len(lm_outputs) > 1 else None
            return outputs

        if features.dim() != 4:
            raise ValueError("features must have shape [B, T, N, C].")
        bsz, time_steps, num_tokens, hidden = features.shape

        if self.language_model is None:
            raise ValueError("language_model is required for VLA forward.")

        if labels is not None and labels.dim() != 3:
            raise ValueError("labels must have shape [B, T, N].")

        # Predict next time-step labels: input t predicts label at t+1.
        if labels is not None:
            if time_steps < 2:
                raise ValueError("Need T >= 2 when labels are provided.")
            input_features = features[:, :-1]
            next_labels = labels[:, 1:]
        else:
            input_features = features
            next_labels = None

        inputs_embeds = input_features.reshape(bsz, -1, hidden)
        flat_labels = None
        if next_labels is not None:
            flat_labels = next_labels.reshape(bsz, -1)

        if instruction_embeds is not None:
            if instruction_embeds.dim() != 3:
                raise ValueError("instruction_embeds must have shape [B, L, C].")
            if instruction_embeds.size(0) != bsz or instruction_embeds.size(2) != hidden:
                raise ValueError("instruction_embeds must match batch and hidden dims.")
            inputs_embeds = torch.cat([instruction_embeds, inputs_embeds], dim=1)
            if flat_labels is not None:
                ignore = torch.full(
                    (bsz, instruction_embeds.size(1)),
                    -100,
                    device=flat_labels.device,
                    dtype=flat_labels.dtype,
                )
                flat_labels = torch.cat([ignore, flat_labels], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.size()[:2],
                device=inputs_embeds.device,
                dtype=torch.long,
            )
        else:
            if attention_mask.dim() != 2 or attention_mask.size(0) != bsz:
                raise ValueError("attention_mask must have shape [B, seq_len].")
            if attention_mask.size(1) != inputs_embeds.size(1):
                if instruction_embeds is not None and attention_mask.size(1) == inputs_embeds.size(1) - instruction_embeds.size(1):
                    prefix = torch.ones(
                        (bsz, instruction_embeds.size(1)),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    )
                    attention_mask = torch.cat([prefix, attention_mask], dim=1)
                else:
                    raise ValueError("attention_mask length does not match inputs_embeds.")

        lm_outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=flat_labels,
            **kwargs
        )
        if isinstance(lm_outputs, dict):
            outputs.update(lm_outputs)
        else:
            outputs["loss"] = lm_outputs[0] if flat_labels is not None else None
            outputs["logits"] = lm_outputs[1] if len(lm_outputs) > 1 else None
        return outputs
    
    def generate_actions(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate actions for given inputs.
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            images: Input images
            **kwargs: Additional arguments
            
        Returns:
            Predicted actions or None if no actions could be generated
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                **kwargs
            )
        return outputs.get('action_logits', None)
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        action_dim: int = 7,
        hidden_dim: int = 768,
        **kwargs
    ) -> "VLAModel":
        """
        Load a pretrained VLA model.
        
        Args:
            model_name_or_path: Path to pretrained model or model identifier
            action_dim: Dimension of action space
            hidden_dim: Hidden dimension
            **kwargs: Additional arguments for model loading
            
        Returns:
            Loaded VLA model
        """
        # This is a placeholder - actual implementation would load from LlamaFactory
        # For now, create an empty model
        model = cls(
            language_model=None,
            vision_encoder=None,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )
        return model
    
    def save_pretrained(self, save_directory: str):
        """
        Save the VLA model.
        
        Args:
            save_directory: Directory to save the model
        """
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # Save config
        config = {
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
        }
        
        import json
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


class _ValueTokenizer:
    def __init__(self, mantissa_max: int = 999, exp_min: int = -6, exp_max: int = 6):
        self.mantissa_max = mantissa_max
        self.exp_min = exp_min
        self.exp_max = exp_max
        self.sign_offset = mantissa_max + 1
        self.mantissa_vocab_size = (mantissa_max + 1) * 2
        self.exponent_vocab_size = (exp_max - exp_min + 1)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        if values.dim() != 3:
            raise ValueError("values must have shape [B, T, D].")
        abs_vals = values.abs()
        is_zero = abs_vals < 1e-8
        exp = torch.floor(torch.log10(abs_vals.clamp_min(1e-8))).to(torch.int64)
        exp = torch.clamp(exp, self.exp_min, self.exp_max)
        scale = torch.pow(10.0, exp.to(values.dtype))
        mantissa = torch.round(abs_vals / scale).to(torch.int64)
        mantissa = torch.clamp(mantissa, 0, self.mantissa_max)
        exp = torch.where(is_zero, torch.zeros_like(exp), exp)
        mantissa = torch.where(is_zero, torch.zeros_like(mantissa), mantissa)
        sign = values < 0
        mantissa_token = mantissa + sign.to(torch.int64) * self.sign_offset
        exponent_token = exp - self.exp_min
        return torch.stack([mantissa_token, exponent_token], dim=-1)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 4 or tokens.size(-1) != 2:
            raise ValueError("tokens must have shape [B, T, D, 2].")
        mantissa_token = tokens[..., 0]
        exponent_token = tokens[..., 1]
        sign = mantissa_token >= self.sign_offset
        mantissa = torch.where(
            sign,
            mantissa_token - self.sign_offset,
            mantissa_token,
        )
        exponent = exponent_token + self.exp_min
        value = mantissa.to(torch.float32) * torch.pow(10.0, exponent.to(torch.float32))
        value = torch.where(sign, -value, value)
        return value
