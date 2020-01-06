import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings

from typing import Any, List, Tuple
try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except:
    warnings.warn("Install apex to improve your performance!")

# pylint:disable=no-member


def gelu(x):
    """Implementation of the gelu activation function.
    """
    return x * 0.5 * (1.0 + torch.erf(x * 0.707106781186547461715))


class CachedBertDecoderEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, past_length, token_type_ids=None):
        position_ids = torch.arange(past_length,
                                    input_ids.shape[-1] + past_length,
                                    dtype=torch.long,
                                    device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class CachedBertEncoderEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, past_length, token_type_ids=None):
        position_ids = torch.arange(past_length,
                                    input_ids.shape[-1] + past_length,
                                    dtype=torch.long,
                                    device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class CachedBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, layer_past, mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # FIX: potential error her
        if layer_past is not None:
            past_key, past_value = layer_past[0], layer_past[1]
            key_layer = torch.cat((past_key, key_layer), dim=-2)
            value_layer = torch.cat((past_value, value_layer), dim=-2)

        present = torch.stack((key_layer, value_layer), dim=0)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.num_attention_heads
        )

        nd, ns = attention_scores.size(-2), attention_scores.size(-1)
        mask = mask[:, :, ns - nd:ns, :ns]

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores * mask - 1e4 * (1 - mask)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)

        # return two tensors
        return context_layer, present


class CachedBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CachedBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = CachedBertSelfAttention(config)
        self.output = CachedBertSelfOutput(config)

    def forward(self, input_tensor, layer_past, mask):
        self_output, present = self.self(input_tensor, layer_past, mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, present


class CachedBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class CachedBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CachedBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = CachedBertAttention(config)
        self.intermediate = CachedBertIntermediate(config)
        self.output = CachedBertOutput(config)

    def forward(self, hidden_states, layer_past, mask):
        attention_output, present = self.attention(hidden_states, layer_past,
                                                   mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, present


class CachedBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([CachedBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, mask,
                past: List) -> Tuple[torch.Tensor, List]:
        presents = []

        for layer_block, layer_past in zip(self.layer, past):
            hidden_states, present = layer_block(hidden_states, layer_past,
                                                 mask)
            presents.append(present)

        return hidden_states, presents


class CachedBertDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = CachedBertDecoderEmbeddings(config)
        self.encoder = CachedBertModel(config)
        self.num_attention_heads = config.num_attention_heads

    def forward(self, input_ids, mask, past=None, past_length=None):
        """
        mask: [batch_size, seq_length] is attention mask
        """
        # Fast way to compute lower triangle attention mask
        mask = mask.to(dtype=torch.uint8)
        mask = mask.view(input_ids.shape[0], 1, 1,
                         -1).expand(input_ids.shape[0], self.num_attention_heads, mask.shape[1],
                                    mask.shape[1])
        mask = (mask + mask.permute(0, 1, 3, 2)) / 2

        # fp16 compatibility
        mask = mask.to(dtype=next(self.parameters()).dtype)
        # lower triangle matrix
        mask = torch.tril(mask)

        # past length calculation and dealing with past
        if past is None:
            past_length = 0
            past = [None] * 12
        else:
            if past_length is None:
                past_length = past[0][0].size(-2)
            else:
                past_length = past_length

        # calculate embedding output
        embedding_output = self.embeddings(input_ids, past_length)

        # Transformer layer
        last_layer_output, presents = self.encoder(embedding_output, mask,
                                                   past)

        return last_layer_output, presents


class CachedBertDecoderLM(nn.Module):
    """Transformer Decoder Language Model
    This module computes the logits output and have a embedding projection matrix.
    """
    def __init__(self, config):
        super().__init__()
        self.transformer = CachedBertDecoder(config)
        self.projection = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.set_tied()

    def set_tied(self):
        """Although weights are set tied explicitly here, when loading new weights,
        call it again.
        """
        self.projection.weight = self.transformer.embeddings.word_embeddings.weight

    def forward(self, input_ids, mask, past=None, past_length=None):
        hidden_states, presents = self.transformer(input_ids, mask, past,
                                                   past_length)
        lm_logits = self.projection(hidden_states)
        return lm_logits, presents


class CachedBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = CachedBertEncoderEmbeddings(config)
        self.encoder = CachedBertModel(config)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads

    def forward(self, input_ids, mask, token_type_ids=None, past=None):
        """
        mask: [batch_size, seq_length] is attention mask
        """
        # Fast way to compute lower triangle attention mask
        mask = mask.to(dtype=torch.uint8)
        mask = mask.view(input_ids.shape[0], 1, 1,
                         -1).expand(input_ids.shape[0], self.num_attention_heads, mask.shape[1],
                                    mask.shape[1])
        mask = (mask + mask.permute(0, 1, 3, 2)) / 2
        # fp16 compatibility
        mask = mask.to(dtype=next(self.parameters()).dtype)

        # past length calculation and dealing with past
        if past is None:
            past_length = 0
            past = [None] * self.num_hidden_layers
        else:
            past_length = past[0][0].size(-2)

        # calculate embedding output
        embedding_output = self.embeddings(input_ids, past_length,
                                           token_type_ids)

        # Transformer layer
        last_layer_output, presents = self.encoder(embedding_output, mask,
                                                   past)

        return last_layer_output, presents