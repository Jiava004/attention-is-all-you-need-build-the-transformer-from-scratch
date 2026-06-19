"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    # TODO: build a token-to-id dict with specials first, then corpus tokens in first-seen order.
    ans = {}
    for tok in specials:
        if tok not in ans:
            ans[tok] = len(ans)
    for words in sentences:
        for tok in words.split(' '):
            if tok not in ans:
                ans[tok] = len(ans)
    return ans

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    # TODO: build the inverse id-to-token dictionary from token_to_id
    return {v: k for k, v in token_to_id.items()}

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    # TODO: convert whitespace tokens of `sentence` to ids via `token_to_id`, using `unk_token`'s id for OOV
    if not sentence:
        return []
    dft_id = token_to_id[unk_token]
    return [token_to_id.get(tok, dft_id) for tok in sentence.split(' ')]

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    # TODO: map each id in ids to its token string via id_to_token and return the list
    return [id_to_token[id] for id in ids]

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    # TODO: return a list of length exactly max_len, padding with pad_id or truncating.
    return (ids + [pad_id] * max_len)[:max_len]

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    # TODO: stack padded id sequences into a (B, L) torch.long tensor
    return torch.tensor(padded_sequences)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    # TODO: rescale embeddings by sqrt(d_model) as in the original Transformer paper
    return embeddings * math.sqrt(d_model)

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    # TODO: return a 1D FloatTensor of length d_model // 2 holding the sinusoidal frequency divisors
    return torch.exp(-math.log(10000) * (torch.arange(0, d_model, 2) / d_model))

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # TODO: build a column vector of position indices from 0 to max_len-1
    return torch.arange(max_len).float().unsqueeze(1)

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    # TODO: write sin(position * div_term) into the even-indexed columns of pe and return it
    pe[:, 0::2] = torch.sin(position * div_term)
    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    # TODO: fill the odd-indexed columns of pe with cos(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    # TODO: build the (max_len, d_model) sinusoidal positional encoding matrix
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(max_len).float().unsqueeze(1)
    div_term = torch.exp(-math.log(10 ** 4) * (torch.arange(0, d_model, 2) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # TODO: add the first L rows of positional_encoding to embedded_batch and return the sum.
    return embedded_batch + positional_encoding[:embedded_batch.shape[1]]

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    # TODO: build a boolean mask marking non-pad positions, shaped for broadcasting against attention scores
    return (token_ids != pad_id)[:, None, None, :]

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    # TODO: build a lower-triangular boolean causal mask of shape (1, 1, seq_len, seq_len)
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))[None, None, :, :]

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    # TODO: combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L).
    return padding_mask & causal_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # TODO: matmul query with the transpose of key over the last two axes
    return query @ key.transpose(-2, -1)

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    # TODO: divide raw attention scores by sqrt(d_k) to stabilize softmax inputs
    return scores / math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # TODO: replace blocked positions of scores with negative infinity
    return scores.masked_fill(~mask, float('-inf'))

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights(masked_scores):
    # TODO: softmax over the last axis, zeroing rows that are entirely -inf
    return torch.nan_to_num(torch.softmax(masked_scores, dim=-1), nan=0.0)

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # TODO: combine attention weights (..., Lq, Lk) with value (..., Lk, d_v)
    return attention_weights @ value

# Step 22 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    # TODO: chain raw scores, scale by sqrt(d_k), optionally mask, softmax, then mix values
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    w = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    return w @ value, w

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # TODO: reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    B, L, d_model = tensor.shape
    return tensor.reshape(B, L, num_heads, d_model // num_heads)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    return split_tensor.transpose(1,2)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    # TODO: merge the head axis back into the feature axis to reconstruct d_model
    B, head, L, d = multi_head_tensor.shape
    return multi_head_tensor.transpose(1,2).reshape(B, L, head * d)

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    ans = x @ weight.transpose(-2, -1)
    if bias is not None:
        ans += bias
    return ans

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # TODO: project x into separate query, key, and value tensors via three linear layers
    return apply_linear_projection(x, w_q, b_q), apply_linear_projection(x, w_k, b_k), apply_linear_projection(x, w_v, b_v)

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple
    q_h = transpose_heads_before_sequence(split_last_dim_into_heads(q, num_heads))
    k_h = transpose_heads_before_sequence(split_last_dim_into_heads(k, num_heads))
    v_h = transpose_heads_before_sequence(split_last_dim_into_heads(v, num_heads))

    return q_h, k_h, v_h

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # TODO: run scaled dot-product attention over per-head Q, K, V and return (context, weights)
    scores = q_h @ k_h.transpose(-1, -2) / math.sqrt(q_h.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    w = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    ctx = w @ v_h
    return ctx, w

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    # TODO: merge the head axis back into d_model and apply the output linear projection.
    return apply_linear_projection(merge_heads_back_to_model_dim(context), w_o, b_o)

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # TODO: project Q/K/V, split into heads, run scaled dot-product attention, merge heads, output projection.
    query = apply_linear_projection(query, w_q, None)
    key = apply_linear_projection(key, w_k, None)
    value = apply_linear_projection(value, w_v, None)
    q_h, k_h, v_h = split_qkv_into_heads(query, key, value, num_heads)
    ctx, w = scaled_dot_product_attention(q_h, k_h, v_h, mask)
    return merge_heads_and_project_output(ctx, w_o, None)

# Step 32 - apply_ffn_first_linear_and_relu
def apply_ffn_first_linear_and_relu(x, w1, b1):
    # TODO: project x by w1, add b1, then apply a ReLU activation.
    return torch.relu(x @ w1 + b1)

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    # TODO: project hidden (..., d_ff) back to (..., d_model) via w2 and b2.
    return hidden @ w2 + b2

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # TODO: compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).
    return apply_ffn_second_linear(apply_ffn_first_linear_and_relu(x, w1, b1), w2, b2)

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    # TODO: return (mean, variance) reduced over the last dim with shape (..., 1)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return mean, var

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # TODO: standardize x along the last axis then apply gamma and beta affine transform
    mean, var = compute_layer_norm_mean_and_variance(x)
    std = (x - mean) / torch.sqrt(var + eps)
    return std * gamma + beta

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # TODO: combine the residual with the sublayer output and layer-normalize the result.
    return normalize_and_scale_with_gamma_beta(residual_input + sublayer_output, gamma, beta, eps)

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    # TODO: multiply x by the boolean keep_mask and rescale by 1/keep_prob.
    return x * keep_mask / keep_prob

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head self-attention on x and wrap with residual add-and-norm.
    sub_layer = assemble_multi_head_attention_forward(x, x, x, w_q, w_k, w_v, w_o, num_heads, src_mask)
    return apply_residual_add_and_norm(x, sub_layer, gamma, beta)

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on x and wrap it with residual add-and-norm.
    sublayer = position_wise_feed_forward_network(x, w1, b1, w2, b2)
    return apply_residual_add_and_norm(x, sublayer, gamma, beta)

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    p = layer_params
    # TODO: chain the self-attention sublayer and the feed-forward sublayer using layer_params.
    # { w_q, w_k, w_v, w_o, attn_gamma, attn_beta, w1, b1, w2, b2, ffn_gamma, ffn_beta } = layer_params
    out = encoder_layer_self_attention_sublayer(x, \
        p['w_q'], p['w_k'], p['w_v'], p['w_o'], p['attn_gamma'], p['attn_beta'], \
        num_heads, src_mask)
    return encoder_layer_feed_forward_sublayer(out, p['w1'], p['b1'], p['w2'], p['b2'], p['ffn_gamma'], p['ffn_beta'])

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    # TODO: sequentially apply each encoder layer to the running hidden state and return the final tensor.
    for layer_param in encoder_layer_params_list:
        x = assemble_encoder_layer(x, layer_param, num_heads, src_mask)
    
    return x

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    # TODO: run masked multi-head self-attention on y and wrap with residual add-and-norm.
    out = assemble_multi_head_attention_forward(y, y, y, w_q, w_k, w_v, w_o, num_heads, tgt_mask)
    return apply_residual_add_and_norm(y, out, gamma, beta)

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head cross-attention (Q from y, K/V from encoder_output) and wrap with add-and-norm
    if src_mask is not None:
        src_mask = src_mask[:,None,None,:]
    out = assemble_multi_head_attention_forward(y, encoder_output, encoder_output, w_q, w_k, w_v, w_o, num_heads, src_mask)
    return apply_residual_add_and_norm(y, out, gamma, beta)

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on y and wrap it with residual add-and-norm
    out = position_wise_feed_forward_network(y, w1, b1, w2, b2)
    return apply_residual_add_and_norm(y, out, gamma, beta)

# Step 46 - assemble_decoder_layer
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN."""
    # TODO: chain the three decoder sublayers using params from layer_params.
    p = layer_params
    
    # 1. 第一步：Masked Self-Attention 子层
    y = decoder_layer_masked_self_attention_sublayer(
        y, 
        p['w_q_self'], p['w_k_self'], p['w_v_self'], p['w_o_self'], 
        p['self_gamma'], p['self_beta'], 
        num_heads, 
        tgt_mask
    )
    
    # 2. 第二步：Cross-Attention 子层
    y = decoder_layer_cross_attention_sublayer(
        y, 
        encoder_output, 
        p['w_q_cross'], p['w_k_cross'], p['w_v_cross'], p['w_o_cross'], 
        p['cross_gamma'], p['cross_beta'], 
        num_heads, 
        src_mask
    )
    
    # 3. 第三步：FFN 子层
    return decoder_layer_feed_forward_sublayer(
        y, 
        p['w1'].T, p['b1'], p['w2'].T, p['b2'], 
        p['ffn_gamma'], p['ffn_beta']
    )

# Step 47 - stack_decoder_layers
import torch
# 彻底覆盖旧函数，在这一关中去掉 .T
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN."""
    p = layer_params
    
    # 1. 第一步：Masked Self-Attention 子层
    y = decoder_layer_masked_self_attention_sublayer(
        y, 
        p['w_q_self'], p['w_k_self'], p['w_v_self'], p['w_o_self'], 
        p['self_gamma'], p['self_beta'], 
        num_heads, 
        tgt_mask
    )
    
    # 2. 第二步：Cross-Attention 子层
    y = decoder_layer_cross_attention_sublayer(
        y, 
        encoder_output, 
        p['w_q_cross'], p['w_k_cross'], p['w_v_cross'], p['w_o_cross'], 
        p['cross_gamma'], p['cross_beta'], 
        num_heads, 
        src_mask
    )
    
    # 3. 第三步：FFN 子层（去掉 .T，直接使用 w1 和 w2）
    return decoder_layer_feed_forward_sublayer(
        y, 
        p['w1'], p['b1'], p['w2'], p['b2'], 
        p['ffn_gamma'], p['ffn_beta']
    )
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    # 循环调用去掉了 .T 的 assemble_decoder_layer
    for layer_param in decoder_layer_params_list:
        y = assemble_decoder_layer(
            y, 
            encoder_output, 
            layer_param, 
            num_heads, 
            src_mask, 
            tgt_mask
        )
    return y

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    # TODO: project decoder hidden states (B, T, D) to vocabulary logits (B, T, V).
    return apply_linear_projection(decoder_output, output_projection_weight, output_projection_bias)

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    return token_embedding_weight.T
    # TODO: return an output projection weight tied to the token embedding matrix

# Step 50 - apply_log_softmax_over_vocab
def apply_log_softmax_over_vocab(logits):
    # TODO: Convert decoder logits (B, T, V) into log probabilities over the vocabulary axis.
    return torch.log_softmax(logits, dim=-1)

# Step 51 - run_transformer_forward
def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    # 0. 基础维度信息
    d_model = model_params['token_embedding'].shape[1]
    src_seq_len = src_ids.shape[1]
    tgt_seq_len = tgt_ids.shape[1]
    
    # 1. 词嵌入 (Embedding Lookup) & 缩放 (Scale)
    src_emb = model_params['token_embedding'][src_ids]
    tgt_emb = model_params['token_embedding'][tgt_ids]
    
    src_scaled = scale_embeddings_by_sqrt_d_model(src_emb, d_model)
    tgt_scaled = scale_embeddings_by_sqrt_d_model(tgt_emb, d_model)
    
    # 2. 注入位置编码 (Sinusoidal Positional Encoding)
    # 取两者中最长的长度生成 PE，防止越界
    max_len = max(src_seq_len, tgt_seq_len)
    pe = build_sinusoidal_positional_encoding(max_len, d_model)
    
    src_pe = add_positional_encoding_to_embeddings(src_scaled, pe)
    tgt_pe = add_positional_encoding_to_embeddings(tgt_scaled, pe)
    
    # 3. 构建掩码 (Masks)
    # 源端 Padding Mask -> 用于 Encoder 自注意力和 Decoder 交叉注意力
    src_mask = build_padding_mask(src_ids, pad_id)
    
    # 目标端 Padding Mask + Causal Mask 结合 -> 用于 Decoder 自注意力
    tgt_pad_mask = build_padding_mask(tgt_ids, pad_id)
    tgt_causal_mask = build_causal_mask(tgt_seq_len)
    tgt_mask = combine_padding_and_causal_masks(tgt_pad_mask, tgt_causal_mask)
    
    # 4. 运行 Encoder 堆叠
    encoder_output = stack_encoder_layers(
        src_pe, 
        model_params['encoder_layers'], 
        num_heads, 
        src_mask
    )
    
    # 5. 运行 Decoder 堆叠
    decoder_output = stack_decoder_layers(
        tgt_pe, 
        encoder_output, 
        model_params['decoder_layers'], 
        num_heads, 
        src_mask, 
        tgt_mask
    )
    
    # 6. 投影到词表大小的对数几率 (Logits)
    logits = apply_final_output_projection(
        decoder_output, 
        model_params['output_projection'], 
        output_projection_bias=None
    )
    
    # 7. 计算对数概率 (Log Softmax) 并返回
    return apply_log_softmax_over_vocab(logits)

# Step 52 - init_encoder_layer_parameters
import torch
import math

def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""
    # TODO: allocate w_q, w_k, w_v, w_o, w1, b1, w2, b2, attn_gamma, attn_beta, ffn_gamma, ffn_beta.
    p = {}
    p['w_q'] = torch.empty(d_model, d_model, dtype=torch.float32)
    p['w_k'] = torch.empty(d_model, d_model, dtype=torch.float32)
    p['w_v'] = torch.empty(d_model, d_model, dtype=torch.float32)
    p['w_o'] = torch.empty(d_model, d_model, dtype=torch.float32)

    p['w1'] = torch.empty(d_model, d_ff, dtype=torch.float32)
    p['w2'] = torch.empty(d_ff, d_model, dtype=torch.float32)

    torch.nn.init.xavier_uniform_(p['w_q'])
    torch.nn.init.xavier_uniform_(p['w_k'])
    torch.nn.init.xavier_uniform_(p['w_v'])
    torch.nn.init.xavier_uniform_(p['w_o'])
    torch.nn.init.xavier_uniform_(p['w1'])
    torch.nn.init.xavier_uniform_(p['w2'])

    p['b1'] = torch.zeros(d_ff, dtype=torch.float32)
    p['b2'] = torch.zeros(d_model, dtype=torch.float32)

    p['attn_gamma'] = torch.ones(d_model, dtype=torch.float32)
    p['ffn_gamma'] = torch.ones(d_model, dtype=torch.float32)
    p['attn_beta'] = torch.zeros(d_model, dtype=torch.float32)
    p['ffn_beta'] = torch.zeros(d_model, dtype=torch.float32)

    for k in p:
        p[k].requires_grad = True

    return p

# Step 53 - init_decoder_layer_parameters
import torch

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    # TODO: return a dict of requires_grad tensors for one decoder layer
    p = {}
    p['w_q_self'] = torch.empty(d_model, d_model, dtype=torch.float32)
    p['w_k_self'] = torch.empty(d_model, d_model, dtype=torch.float32)
    p['w_v_self'] = torch.empty(d_model, d_model, dtype=torch.float32)
    p['w_o_self'] = torch.empty(d_model, d_model, dtype=torch.float32)
    p['w_q_cross'] = torch.empty(d_model, d_model, dtype=torch.float32)
    p['w_k_cross'] = torch.empty(d_model, d_model, dtype=torch.float32)
    p['w_v_cross'] = torch.empty(d_model, d_model, dtype=torch.float32)
    p['w_o_cross'] = torch.empty(d_model, d_model, dtype=torch.float32)

    p['w1'] = torch.empty(d_model, d_ff, dtype=torch.float32)
    p['w2'] = torch.empty(d_ff, d_model, dtype=torch.float32)

    torch.nn.init.xavier_uniform_(p['w_q_self'])
    torch.nn.init.xavier_uniform_(p['w_k_self'])
    torch.nn.init.xavier_uniform_(p['w_v_self'])
    torch.nn.init.xavier_uniform_(p['w_o_self'])
    torch.nn.init.xavier_uniform_(p['w_q_cross'])
    torch.nn.init.xavier_uniform_(p['w_k_cross'])
    torch.nn.init.xavier_uniform_(p['w_v_cross'])
    torch.nn.init.xavier_uniform_(p['w_o_cross'])
    torch.nn.init.xavier_uniform_(p['w1'])
    torch.nn.init.xavier_uniform_(p['w2'])

    p['b1'] = torch.zeros(d_ff, dtype=torch.float32)
    p['b2'] = torch.zeros(d_model, dtype=torch.float32)

    p['self_gamma'] = torch.ones(d_model, dtype=torch.float32)
    p['self_beta'] = torch.zeros(d_model, dtype=torch.float32)
    p['cross_gamma'] = torch.ones(d_model, dtype=torch.float32)
    p['cross_beta'] = torch.zeros(d_model, dtype=torch.float32)

    p['ffn_gamma'] = torch.ones(d_model, dtype=torch.float32)
    p['ffn_beta'] = torch.zeros(d_model, dtype=torch.float32)

    for k in p:
        p[k].requires_grad = True

    return p

# Step 54 - init_embedding_and_projection_parameters
import torch

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    # TODO: allocate three (vocab_size, d_model) tensors with requires_grad=True
    p = {}
    p['src_embedding'] = torch.randn(vocab_size, d_model, dtype=torch.float32)
    p['tgt_embedding'] = torch.randn(vocab_size, d_model, dtype=torch.float32)
    p['output_projection'] = torch.randn(vocab_size, d_model, dtype=torch.float32) if not tie_weights else p['tgt_embedding']

    for k in p:
        p[k].requires_grad = True

    return p

# Step 55 - collect_model_parameters_into_list
import torch

def collect_model_parameters_into_list(encoder_layer_params, decoder_layer_params, embedding_params):
    # TODO: walk the encoder, decoder, and embedding dicts and return a flat deduped list of tensors
    params = []
    seen_ids = set()
    
    # 辅助函数：如果是 Tensor 且没被收集过，则加入列表
    def add_tensor(t):
        if isinstance(t, torch.Tensor):
            t_id = id(t)
            if t_id not in seen_ids:
                seen_ids.add(t_id)
                params.append(t)
    # 1. 遍历所有的 Encoder 层字典（encoder_layer_params 是一个 list，里面每个元素是 dict）
    for layer in encoder_layer_params:
        for val in layer.values():
            add_tensor(val)
            
    # 2. 遍历所有的 Decoder 层字典（decoder_layer_params 也是 list 嵌套 dict）
    for layer in decoder_layer_params:
        for val in layer.values():
            add_tensor(val)
            
    # 3. 遍历 Embedding 字典（embedding_params 是单个 dict）
    for val in embedding_params.values():
        add_tensor(val)
        
    return params

# Step 56 - shift_targets_right_with_start_token
def shift_targets_right_with_start_token(target_ids, start_token_id):
    # TODO: prepend start_token_id and drop the last column so output shape matches target_ids
    out = torch.empty_like(target_ids)
    out[:, 0] = start_token_id
    out[:, 1:] = target_ids[:, :-1]
    return out

# Step 57 - compute_noam_learning_rate (not yet solved)
# TODO: implement

# Step 58 - build_uniform_smoothing_distribution (not yet solved)
# TODO: implement

# Step 59 - set_confidence_on_gold_tokens (not yet solved)
# TODO: implement

# Step 60 - zero_pad_column_and_pad_token_rows (not yet solved)
# TODO: implement

# Step 61 - compute_label_smoothed_kl_loss (not yet solved)
# TODO: implement

# Step 62 - average_loss_over_non_pad_tokens (not yet solved)
# TODO: implement

# Step 63 - compute_token_accuracy_ignoring_pad (not yet solved)
# TODO: implement

# Step 64 - initialize_adam_optimizer_state (not yet solved)
# TODO: implement

# Step 65 - update_adam_first_moment (not yet solved)
# TODO: implement

# Step 66 - update_adam_second_moment (not yet solved)
# TODO: implement

# Step 67 - apply_adam_bias_correction (not yet solved)
# TODO: implement

# Step 69 - apply_adam_step_to_all_parameters (not yet solved)
# TODO: implement

# Step 70 - zero_all_parameter_gradients (not yet solved)
# TODO: implement

# Step 71 - compute_batch_training_loss (not yet solved)
# TODO: implement

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

