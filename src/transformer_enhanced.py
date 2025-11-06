import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils import checkpoint # 确保导入

# ==============================
# 相对位置偏置（T5-style）
# (已修复设备问题)
# ==============================
class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_dist=128):
        super().__init__()
        self.num_heads = num_heads
        self.max_dist = max_dist
        self.rel_pos_bias = nn.Embedding(2 * max_dist + 1, num_heads)

    def forward(self, seq_len):
        device = self.rel_pos_bias.weight.device
        context_position = torch.arange(seq_len, device=device)[:, None]
        memory_position = torch.arange(seq_len, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position = torch.clamp(relative_position, -self.max_dist, self.max_dist)
        relative_position += self.max_dist
        bias = self.rel_pos_bias(relative_position)
        return bias.permute(2, 0, 1).unsqueeze(0)

# ==============================
# 线性注意力（Softmax-free）
# [!!! 最终修复 !!!]
# ==============================
def linear_attention(q, k, v, mask=None):
    """更稳定的线性注意力实现（AMP 兼容）。
    - 内部在 float32 上进行计算，避免 FP16/AMP 下的数值不稳定。
    - [FIX] 使用 softmax feature map 替代 elu(x)+1 来保证数值稳定性
    """
    orig_dtype = q.dtype
    q = q.float()
    k = k.float()
    v = v.float()

    # --- [FIX] 使用 softmax 保证 q, k > 0 且和为 1，防止数值爆炸 ---
    q = torch.softmax(q, dim=-1)
    k = torch.softmax(k, dim=-1)
    # --- [FIX END] ---

    if mask is not None:
        key_mask = mask
        # 统一处理不同形状的 mask 到 [B, 1, K, 1]
        if key_mask.dim() == 4:
            if key_mask.size(-2) == 1:
                key_mask = key_mask.squeeze(-2).squeeze(1)  # [B, K]
            else:
                key_mask = key_mask.squeeze(1)
                key_mask = torch.diagonal(key_mask, dim1=-2, dim2=-1)  # [B/1, T]
        elif key_mask.dim() == 3:
            if key_mask.size(1) == 1:
                key_mask = key_mask.squeeze(1)  # [B, K]
            else:
                key_mask = torch.diagonal(key_mask, dim1=-2, dim2=-1)  # [B, T]

        key_mask = key_mask.to(dtype=torch.bool)
        key_mask = key_mask[:, None, :, None]  # [B, 1, K, 1]

        k = k.masked_fill(~key_mask, 0.0)
        v = v.masked_fill(~key_mask, 0.0)

    kv = torch.einsum("bhld,bhle->bhde", k, v)
    k_sum = k.sum(dim=-2) 
    
    # [FIX] 彻底防止 0/0
    # 我们给分母加上一个很小的值，而不是 clamp()
    # clamp() 在 fp16 下如果分子分母都极小，仍可能导致 0/0 = NaN
    denominator = torch.einsum("bhld,bhd->bhl", q, k_sum) + 1e-6 
    
    numerator = torch.einsum("bhld,bhde->bhle", q, kv)

    out = numerator / denominator.unsqueeze(-1) # <--- 现在是安全的
    
    return out.to(orig_dtype)

# ==============================
# 多头注意力（支持相对位置 + 线性注意力）
# [!!! 最终修复 !!!]
# ==============================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1,
                 use_relative=False, max_rel_dist=128, use_linear=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_relative = use_relative
        self.use_linear = use_linear
        if use_relative:
            self.rel_bias = RelativePositionBias(num_heads, max_rel_dist)

    def forward(self, q, k, v, mask=None):
        bs, q_len, _ = q.shape
        _, k_len, _ = k.shape

        q = self.q_linear(q).view(bs, q_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, k_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, k_len, self.num_heads, self.d_k).transpose(1, 2)

        use_linear_effective = self.use_linear
        if use_linear_effective and mask is not None and mask.dim() == 4:
            is_square = (mask.size(-2) == mask.size(-1)) and (mask.size(-2) > 1)
            if is_square:
                use_linear_effective = False # (这是你的正确设计，保留)

        if use_linear_effective:
            scale = 1.0 / math.sqrt(self.d_k)
            # [FIX] linear_attention 现在是数值稳定的
            context = linear_attention(q * scale, k * scale, v, mask) 
            
            # [!!! 关键修复 !!!]
            # 删除 nan_to_num，它会隐藏 NaN 并导致梯度失败
            # context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
            # --- [修复结束] ---
            
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            if self.use_relative and q_len == k_len:
                scores = scores + self.rel_bias(q_len).to(scores.device)
            if mask is not None:
                fill_val = torch.finfo(scores.dtype).min
                scores = scores.masked_fill(mask == False, fill_val)
            attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
            attn = self.dropout(attn)
            context = torch.matmul(attn, v)

        context = context.transpose(1, 2).contiguous().view(bs, q_len, -1)
        return self.out_linear(context)

# ==============================
# 前馈网络
# (无修改)
# ==============================
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)

# ==============================
# 绝对位置编码（可选）
# (无修改)
# ==============================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ==============================
# Encoder Layer (Pre-Norm)
# (已修复为 Pre-Norm)
# ==============================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1,
                 use_relative=False, use_linear=False, use_checkpointing=True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout,
                                          use_relative=use_relative, use_linear=use_linear)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.use_checkpointing = use_checkpointing

    def forward(self, x, mask=None):
        
        def create_attn(args):
            x_norm_attn, mask_attn = args
            return self.self_attn(x_norm_attn, x_norm_attn, x_norm_attn, mask_attn)

        x_norm1 = self.norm1(x) # 1. Pre-Norm
        if self.use_checkpointing and self.training:
            attn_out = checkpoint.checkpoint(create_attn, (x_norm1, mask), use_reentrant=False)
        else:
            attn_out = create_attn((x_norm1, mask))
        x = x + self.dropout1(attn_out) # 2. Add


        def create_ff(args):
            x_norm_ff = args
            return self.ff(x_norm_ff)

        x_norm2 = self.norm2(x) # 3. Pre-Norm
        if self.use_checkpointing and self.training:
            ff_out = checkpoint.checkpoint(create_ff, x_norm2, use_reentrant=False)
        else:
            ff_out = create_ff(x_norm2)
        return x + self.dropout2(ff_out) # 4. Add

# ==============================
# Decoder Layer (Pre-Norm)
# (已修复为 Pre-Norm)
# ==============================
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1,
                 use_relative=False, use_linear=False, use_checkpointing=True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout,
                                          use_relative=use_relative, use_linear=use_linear)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout,
                                           use_relative=False, use_linear=use_linear)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.use_checkpointing = use_checkpointing

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        
        # 1. Self-Attention
        def self_attn_fn(args):
            x_norm_attn, mask_attn = args
            return self.self_attn(x_norm_attn, x_norm_attn, x_norm_attn, mask_attn)
        
        x_norm1 = self.norm1(x) # Pre-Norm
        if self.use_checkpointing and self.training:
            self_attn_out = checkpoint.checkpoint(self_attn_fn, (x_norm1, tgt_mask), use_reentrant=False)
        else:
            self_attn_out = self_attn_fn((x_norm1, tgt_mask))
        x = x + self.dropout1(self_attn_out) # Add

        
        # 2. Cross-Attention
        def cross_attn_fn(args):
            x_norm_cross, enc_out_attn, mask_attn = args
            return self.cross_attn(x_norm_cross, enc_out_attn, enc_out_attn, mask_attn)

        x_norm2 = self.norm2(x) # Pre-Norm
        if self.use_checkpointing and self.training:
            cross_attn_out = checkpoint.checkpoint(cross_attn_fn, (x_norm2, enc_out, src_mask), use_reentrant=False)
        else:
            cross_attn_out = cross_attn_fn((x_norm2, enc_out, src_mask))
        x = x + self.dropout2(cross_attn_out) # Add

        
        # 3. FeedForward
        def create_ff(args):
            x_norm_ff = args
            return self.ff(x_norm_ff)

        x_norm3 = self.norm3(x) # Pre-Norm
        if self.use_checkpointing and self.training:
            ff_out = checkpoint.checkpoint(create_ff, x_norm3, use_reentrant=False)
        else:
            ff_out = create_ff(x_norm3)
        return x + self.dropout3(ff_out) # Add


# ==============================
# 完整 Transformer (Pre-Norm, Residual Init)
# ==============================
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8,
                 d_ff=2048, max_src_len=512, max_tgt_len=128, dropout=0.1,
                 use_relative=False, use_linear=False,
                 use_absolute_pe=True, use_checkpointing=True,
                 pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.use_absolute_pe = use_absolute_pe
        if self.use_absolute_pe:
            self.src_pos_enc = PositionalEncoding(d_model, max_src_len)
            self.tgt_pos_enc = PositionalEncoding(d_model, max_tgt_len)

        self.dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout,
                         use_relative=use_relative, use_linear=use_linear,
                         use_checkpointing=use_checkpointing)
            for _ in range(num_layers)
        ])
        
        self.encoder_norm = nn.LayerNorm(d_model) # Pre-Norm 最终 Norm
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout,
                         use_relative=use_relative, use_linear=use_linear,
                         use_checkpointing=use_checkpointing)
            for _ in range(num_layers)
        ])
        
        self.decoder_norm = nn.LayerNorm(d_model) # Pre-Norm 最终 Norm
        
        self.out_linear = nn.Linear(d_model, vocab_size)
        
        self.num_layers = num_layers # 保存 N
        
        self._init_weights() # 调用残差初始化

    # (已修复为 Pre-Norm 的残差初始化)
    def _init_weights(self):
        std = 0.02
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

        # 残差初始化: 缩放 MHA out 和 FFN linear2
        residual_std = std / math.sqrt(2 * self.num_layers) 
        
        for name, module in self.named_modules():
            if isinstance(module, MultiHeadAttention):
                nn.init.normal_(module.out_linear.weight, mean=0.0, std=residual_std)
            elif isinstance(module, FeedForward):
                nn.init.normal_(module.linear2.weight, mean=0.0, std=residual_std)
    
    # (已修复为正确的组合掩码)
    def forward(self, src, tgt):
        # 1. 源掩码 (Padding Mask) [B, 1, 1, T_src]
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

        # 2. 目标掩码 (Padding Mask + Causal Mask)
        tgt_len = tgt.size(1)
        # 2a. 因果掩码 [1, 1, T_tgt, T_tgt]
        causal_mask = torch.tril(
            torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool)
        ).unsqueeze(0).unsqueeze(0)
        # 2b. 目标填充掩码 [B, 1, 1, T_tgt]
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # 2c. 组合掩码 [B, 1, T_tgt, T_tgt]
        tgt_mask = tgt_pad_mask & causal_mask
        
        # (Embedding & PE)
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        if self.use_absolute_pe:
            src_emb = self.src_pos_enc(src_emb)
            tgt_emb = self.tgt_pos_enc(tgt_emb)
        src_emb = self.dropout(src_emb)
        tgt_emb = self.dropout(tgt_emb)

        # (Encoder 栈)
        enc_out = src_emb
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)
        enc_out = self.encoder_norm(enc_out) # (Pre-Norm 最终 Norm)

        # (Decoder 栈)
        dec_out = tgt_emb
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, tgt_mask, src_mask)
        dec_out = self.decoder_norm(dec_out) # (Pre-Norm 最终 Norm)

        return self.out_linear(dec_out)