# # Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# # Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

# """A self-contained, highly-configurable, modernized BERT model for Megatron-LM."""

# import torch
# import torch.nn.functional as F
# import numbers
# import inspect

# from megatron import get_args
# from megatron.core import tensor_parallel
# from megatron.core.utils import make_viewless_tensor
# from megatron.model.module import MegatronModule
# from megatron.model.language_model import parallel_lm_logits
# from megatron.model.utils import get_linear_layer, init_method_normal
# # [修正] 標準的なLayerNormも使う可能性があるのでimportしておく
# from megatron.model import LayerNorm

# from deepspeed.runtime.zero import GatheredParameters

# # Apexのimport部分はそのまま
# try:
#     from apex.contrib.layer_norm.layer_norm import FastLayerNormFN
#     from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction
#     HAVE_APEX = True
# except ImportError:
#     print("Warning: Apex is not installed. High-performance LayerNorm will not be available.")
#     HAVE_APEX = False

# if not HAVE_APEX: raise ImportError("Apex is required to use MixedFusedLayerNorm.")

# # flash-attnのimport部分はそのまま
# try:
#     from flash_attn import flash_attn_func
# except ImportError:
#     print("Warning: flash-attn is not installed. It is required for HybridFlashAttention.")
#     flash_attn_func = None
    
# from megatron.model.rotary_pos_embedding import RotaryEmbedding, apply_rotary_pos_emb

# def backward_hook_checker(module, grad_input, grad_output):
#     module_name = module.__class__.__name__
#     print(f"\n>>> Backward hook fired for: [{module_name}]")
    
#     for i, grad in enumerate(grad_output):
#         if grad is None:
#             print(f"  - grad_output[{i}] is None! <--- check this grad！")
#         else:
#             print(f"  - grad_output[{i}].shape: {grad.shape}, grad.mean(): {grad.abs().mean():.2e}, grad.std(): {grad.std():.2e}")


# # inspired by megatron.core.fused_layer_norm
# class BiaslessMixedFusedLayerNorm(torch.nn.Module):
#     def __init__(self, normalized_shape, eps=1e-5, sequence_parallel=False, no_persist_layer_norm=True, mem_efficient_ln=True):
#         super().__init__()
#         if not HAVE_APEX: raise ImportError("Apex is required to use MixedFusedLayerNorm.")
#         persist_ln_hidden_sizes = [1024, 1536, 2048, 2304, 3072, 3840, 4096, 5120, 6144, 8192, 10240, 12288, 12800, 15360, 16384, 18432, 20480, 24576, 25600, 30720, 32768, 40960, 49152, 65536]
#         if normalized_shape not in persist_ln_hidden_sizes: no_persist_layer_norm = True
#         if isinstance(normalized_shape, numbers.Integral): normalized_shape = (normalized_shape,)
#         self.normalized_shape = torch.Size(normalized_shape)
#         self.eps = eps
#         self.sequence_parallel = sequence_parallel
#         self.no_persist_layer_norm = no_persist_layer_norm
#         self.mem_efficient_ln = mem_efficient_ln
#         self.weight = torch.nn.Parameter(torch.empty(*self.normalized_shape, dtype=get_args().params_dtype))
#         self.register_buffer('bias', torch.zeros(*self.normalized_shape, dtype=get_args().params_dtype), persistent=False)
#         self.reset_parameters()
#         setattr(self.weight, 'sequence_parallel', self.sequence_parallel)
#     def reset_parameters(self):
#         torch.nn.init.ones_(self.weight)
#     def forward(self, input):
#         if not input.is_cuda: return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
#         if self.no_persist_layer_norm:
#             if 'memory_efficient' in inspect.getfullargspec(FusedLayerNormAffineFunction.forward).args:
#                 return FusedLayerNormAffineFunction.apply(input, self.weight, self.bias, self.normalized_shape, self.eps, self.mem_efficient_ln)
#             else:
#                 return FusedLayerNormAffineFunction.apply(input, self.weight, self.bias, self.normalized_shape, self.eps)
#         else:
#             output = FastLayerNormFN.apply(input, self.weight, self.bias, self.eps)
#             return make_viewless_tensor(inp=output, requires_grad=input.requires_grad, keep_graph=True)


# class GeGLUMLP(MegatronModule):
#     def __init__(self, config):
#         super().__init__()
#         self.dense_h_to_4h_gated = tensor_parallel.ColumnParallelLinear(config.hidden_size, config.ffn_hidden_size * 2, config=config, init_method=config.init_method, bias=False)
#         self.dense_4h_to_h = tensor_parallel.RowParallelLinear(config.ffn_hidden_size, config.hidden_size, config=config, init_method=config.output_layer_init_method, bias=False, input_is_parallel=True)
#     def forward(self, hidden_states):
#         gated_output, _ = self.dense_h_to_4h_gated(hidden_states)
#         x1, x2 = torch.chunk(gated_output, 2, dim=-1)
#         intermediate = F.gelu(x1) * x2
#         output, _ = self.dense_4h_to_h(intermediate)
#         return output

# class HybridFlashAttention(MegatronModule):
#     def __init__(self, config, is_global):
#         super().__init__()
#         if flash_attn_func is None: raise RuntimeError("flash-attn is not installed.")
#         self.config = config
#         self.is_global = is_global
#         self.local_window_size = getattr(config, 'local_window_size', 256)
#         self.qkv_proj = tensor_parallel.ColumnParallelLinear(config.hidden_size, config.hidden_size * 3, config=config, init_method=config.init_method, bias=False)
#         self.dense = tensor_parallel.RowParallelLinear(config.hidden_size, config.hidden_size, config=config, init_method=config.output_layer_init_method, bias=False, input_is_parallel=True)

#     def forward(self, hidden_states, rotary_pos_emb):
#         s, b = hidden_states.size(0), hidden_states.size(1)
#         mixed_qkv_layer, _ = self.qkv_proj(hidden_states)
#         (query_layer, key_layer, value_layer) = torch.split(
#             mixed_qkv_layer, self.config.hidden_size, dim=-1)
#         nh = self.config.num_attention_heads
#         hd = self.config.kv_channels
#         new_shape = (s, b, nh, hd)
#         query_layer = query_layer.view(new_shape)
#         key_layer = key_layer.view(new_shape)
#         value_layer = value_layer.view(new_shape)
        
#         # [デバッグ修正] RoPEの適用を一時的に無効化し、原因を調査します
#         # cos_emb, _ = rotary_pos_emb
#         # if cos_emb.dim() == 4:
#         #     cos_emb = cos_emb.squeeze(1).squeeze(1)
#         # freqs = torch.acos(cos_emb)
#         # freqs = freqs.unsqueeze(1).unsqueeze(1)
#         # query_layer = apply_rotary_pos_emb(query_layer, freqs)
#         # key_layer = apply_rotary_pos_emb(key_layer, freqs)

#         query_layer = query_layer.transpose(0, 1)
#         key_layer = key_layer.transpose(0, 1)
#         value_layer = value_layer.transpose(0, 1)
        
#         attn_output = F.scaled_dot_product_attention(
#             query_layer, 
#             key_layer, 
#             value_layer, 
#             attn_mask=None,
#             is_causal=False
#         )
#         attn_output = attn_output.transpose(0, 1).contiguous()
#         attn_output = attn_output.reshape(hidden_states.shape)
#         output, _ = self.dense(attn_output)
#         return output


# class ModernTransformerLayer(MegatronModule):
#     def __init__(self, config, layer_number, global_attn_every_n_layers):
#         super().__init__()
#         self.input_layernorm = BiaslessMixedFusedLayerNorm(config.hidden_size, eps=config.layernorm_epsilon, sequence_parallel=config.sequence_parallel)
#         self.post_attention_layernorm = BiaslessMixedFusedLayerNorm(config.hidden_size, eps=config.layernorm_epsilon, sequence_parallel=config.sequence_parallel)
#         self.is_global = (layer_number + 1) % global_attn_every_n_layers == 0
#         self.attention = HybridFlashAttention(config, is_global=self.is_global)
#         self.mlp = GeGLUMLP(config)

#     def forward(self, hidden_states, rotary_pos_emb_global, rotary_pos_emb_local):
#         rotary_to_use = rotary_pos_emb_global if self.is_global else rotary_pos_emb_local
#         residual = hidden_states
#         normalized_states = self.input_layernorm(hidden_states)
#         attention_output = self.attention(normalized_states, rotary_to_use)
#         hidden_states = residual + attention_output
#         residual = hidden_states
#         normalized_states = self.post_attention_layernorm(hidden_states)
#         mlp_output = self.mlp(normalized_states)
#         hidden_states = residual + mlp_output
#         return hidden_states

# # get_linear_layer_custom はそのまま
# def get_linear_layer_custom(rows, columns, init_method, bias=True, gather_params_on_init=False):
#     layer = torch.nn.Linear(rows, columns, bias=bias)
#     args = get_args()
#     if args.perform_initialization:
#         with GatheredParameters(layer.weight, modifier_rank=0, enabled=gather_params_on_init):
#             init_method(layer.weight)
#     if bias:
#         with torch.no_grad():
#             with GatheredParameters(layer.bias, modifier_rank=0, enabled=gather_params_on_init):
#                 layer.bias.zero_()
#     return layer

# class ModernBertLMHead(MegatronModule):
#     def __init__(self, mpu_vocab_size, hidden_size, config, parallel_output):
#         super().__init__(config=config)
#         self.parallel_output = parallel_output
#         self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
#         tensor_parallel.set_tensor_model_parallel_attributes(self.bias, True, 0, 1)
#         self.dense = get_linear_layer_custom(hidden_size, hidden_size, config.init_method, bias=False)
#         self.layernorm = BiaslessMixedFusedLayerNorm(hidden_size, eps=config.layernorm_epsilon, sequence_parallel=config.sequence_parallel)
#         self.gelu = F.gelu
#     def forward(self, hidden_states, word_embeddings_weight):
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.gelu(hidden_states)
#         hidden_states = self.layernorm(hidden_states)
#         output = parallel_lm_logits(hidden_states, word_embeddings_weight, self.parallel_output, bias=self.bias)
#         return output


# class ModernBertModel(MegatronModule):
#     def __init__(self, config, num_tokentypes=2, parallel_output=True):
#         super().__init__(config=config)
#         args = get_args()
#         self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        
#         self.word_embeddings = tensor_parallel.VocabParallelEmbedding(args.padded_vocab_size, config.hidden_size, config=config, init_method=config.init_method)
#         self.embedding_layernorm = BiaslessMixedFusedLayerNorm(config.hidden_size, eps=config.layernorm_epsilon, sequence_parallel=config.sequence_parallel)
#         self.embedding_dropout = torch.nn.Dropout(config.hidden_dropout)

#         rotary_dim = config.kv_channels or (config.hidden_size // config.num_attention_heads)
#         self.rotary_pos_emb_global = RotaryEmbedding(rotary_dim, theta=args.global_rope_theta)
#         self.rotary_pos_emb_local = RotaryEmbedding(rotary_dim, theta=args.local_rope_theta)
        
#         self.transformer_layers = torch.nn.ModuleList(
#             [ModernTransformerLayer(config, i, args.global_attn_every_n_layers) for i in range(config.num_layers)]
#         )
#         self.lm_head = ModernBertLMHead(args.padded_vocab_size, config.hidden_size, config, parallel_output)
        
#         # Backward hook はデバッグ用にコメントアウトされたままにしておきます
#         # self.register_full_backward_hook(backward_hook_checker)

#     def set_input_tensor(self, input_tensor):
#         pass

#     def forward(self, input_ids, tokentype_ids=None, lm_labels=None):
#         # ステップ1: Embedding層と最初のLayerNorm
#         word_embeds = self.word_embeddings(input_ids)
#         embeddings = word_embeds
#         hidden_states = self.embedding_layernorm(embeddings)
#         hidden_states = self.embedding_dropout(hidden_states)

#         hidden_states = hidden_states.transpose(0, 1).contiguous()
        
#         seq_len = hidden_states.size(0)
#         rotary_pos_emb_global = self.rotary_pos_emb_global(seq_len)
#         rotary_pos_emb_local = self.rotary_pos_emb_local(seq_len)
        
#         # =================================================================
#         # >>>>>>>>>>>>>>>>>>>> [LayerNorm2回問題の修正] <<<<<<<<<<<<<<<<<<<<
#         # =================================================================
#         # 1層目の処理をループの外に出し、input_layernorm をスキップする
        
#         first_layer = self.transformer_layers[0]
#         rotary_to_use = rotary_pos_emb_global if first_layer.is_global else rotary_pos_emb_local
        
#         # 1層目のAttentionブロック
#         residual = hidden_states
#         # `embedding_layernorm` の出力をそのまま使う (first_layer.input_layernormをスキップ)
#         attention_output = first_layer.attention(hidden_states, rotary_to_use)
#         hidden_states = residual + attention_output
        
#         # 1層目のMLPブロック
#         residual = hidden_states
#         normalized_states = first_layer.post_attention_layernorm(hidden_states)
#         mlp_output = first_layer.mlp(normalized_states)
#         hidden_states = residual + mlp_output

#         # 2層目以降の処理 (通常通り、内部でinput_layernormが適用される)
#         if len(self.transformer_layers) > 1:
#             for layer in self.transformer_layers[1:]:
#                 hidden_states = layer(hidden_states, rotary_pos_emb_global, rotary_pos_emb_local)
#         # =================================================================
        
#         # ステップ4: LM Head
#         lm_logits = self.lm_head(hidden_states, self.word_embeddings.weight)

#         # ステップ5: loss計算 or logits返却 (この部分は変更なし)
#         if lm_labels is None:
#             return lm_logits.transpose(0, 1).contiguous()
#         else:
#             lm_labels_s_b = lm_labels.transpose(0, 1).contiguous()
#             if self.fp16_lm_cross_entropy:
#                 assert lm_logits.dtype == torch.half
#                 loss_per_token_s_b = tensor_parallel.vocab_parallel_cross_entropy(
#                     lm_logits, lm_labels_s_b)
#             else:
#                 loss_per_token_s_b = tensor_parallel.vocab_parallel_cross_entropy(
#                     lm_logits.float(), lm_labels_s_b)
#             loss_per_token_b_s = loss_per_token_s_b.transpose(0, 1).contiguous()
#             return loss_per_token_b_s

#     def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
#         return super().state_dict(prefix=prefix, keep_vars=keep_vars)
    
#     def load_state_dict(self, state_dict, strict=True):
#         super().load_state_dict(state_dict, strict=strict)

# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""
[デバッグ用・標準BERT構成版]
クラス名はそのままに、内部を一般的なBERTのコンポーネントに置き換えたモデル。
"""

import torch
import torch.nn.functional as F
import numbers

from megatron import get_args
from megatron.core import tensor_parallel
from megatron.model.module import MegatronModule
from megatron.model.language_model import parallel_lm_logits
from megatron.model import LayerNorm # 標準のLayerNormをメインで使用

# --- 各クラスの「中身」を標準的な実装に置き換えます ---

class BiaslessMixedFusedLayerNorm(MegatronModule):
    def __init__(self, normalized_shape, eps=1e-5, sequence_parallel=False, **kwargs):
        super().__init__()
        self.layer_norm = LayerNorm(normalized_shape, eps=eps, sequence_parallel=sequence_parallel)

    def forward(self, input):
        return self.layer_norm(input)


class GeGLUMLP(MegatronModule):
    """
    [修正] クラス名はそのままに、内部を標準的なMLP(GELU)に変更。
    """
    def __init__(self, config):
        super().__init__(config=config)
        self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
            config.hidden_size, config.ffn_hidden_size,
            config=config, init_method=config.init_method, bias=True)
        self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
            config.ffn_hidden_size, config.hidden_size,
            config=config, init_method=config.output_layer_init_method, bias=True, input_is_parallel=True)

    def forward(self, hidden_states):
        intermediate, _ = self.dense_h_to_4h(hidden_states)
        intermediate = F.gelu(intermediate)
        output, _ = self.dense_4h_to_h(intermediate)
        return output


class HybridFlashAttention(MegatronModule):
    """
    [修正] クラス名はそのままに、内部を標準的なSelf-Attentionに変更 (次元修正版)
    """
    def __init__(self, config, **kwargs):
        super().__init__(config=config)
        self.qkv_proj = tensor_parallel.ColumnParallelLinear(
            config.hidden_size, 3 * config.hidden_size,
            config=config, init_method=config.init_method, bias=True)
        self.dense = tensor_parallel.RowParallelLinear(
            config.hidden_size, config.hidden_size,
            config=config, init_method=config.output_layer_init_method, bias=True, input_is_parallel=True)

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [s, b, h] (シーケンス長, バッチサイズ, 隠れ層サイズ)
        s, b, h = hidden_states.shape
        mixed_qkv_layer, _ = self.qkv_proj(hidden_states)
        (query_layer, key_layer, value_layer) = torch.split(
            mixed_qkv_layer, self.config.hidden_size, dim=-1)

        nh = self.config.num_attention_heads
        hd = self.config.hidden_size // nh
        
        # テンソルの形状を (s, b, nh, hd) に変形
        query_layer = query_layer.view(s, b, nh, hd)
        key_layer = key_layer.view(s, b, nh, hd)
        value_layer = value_layer.view(s, b, nh, hd)

        # [修正] scaled_dot_product_attentionが期待する (b, nh, s, hd) の形状に並べ替え
        query_layer = query_layer.permute(1, 2, 0, 3)
        key_layer = key_layer.permute(1, 2, 0, 3)
        value_layer = value_layer.permute(1, 2, 0, 3)

        # attention_mask は [b, 1, s, s] の形状を期待される
        context_layer = F.scaled_dot_product_attention(
            query_layer, key_layer, value_layer,
            attn_mask=attention_mask, is_causal=False)

        # [修正] 元の (s, b, h) の形状に戻すための並べ替えと変形
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        context_layer = context_layer.view(s, b, h)
        
        output, _ = self.dense(context_layer)
        return output


class ModernTransformerLayer(MegatronModule):
    """
    [修正] 標準部品を使うように内部を修正。forwardの引数も変更。
    """
    def __init__(self, config, layer_number, global_attn_every_n_layers):
        super().__init__(config=config)
        self.input_layernorm = BiaslessMixedFusedLayerNorm(config.hidden_size, eps=config.layernorm_epsilon, sequence_parallel=config.sequence_parallel)
        self.post_attention_layernorm = BiaslessMixedFusedLayerNorm(config.hidden_size, eps=config.layernorm_epsilon, sequence_parallel=config.sequence_parallel)
        self.attention = HybridFlashAttention(config)
        self.mlp = GeGLUMLP(config)

    def forward(self, hidden_states, attention_mask):
        # Attention Block (Pre-LN)
        residual = hidden_states
        layernorm_output = self.input_layernorm(hidden_states)
        attention_output = self.attention(layernorm_output, attention_mask)
        hidden_states = residual + attention_output
        
        # MLP Block (Pre-LN)
        residual = hidden_states
        layernorm_output = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(layernorm_output)
        hidden_states = residual + mlp_output
        return hidden_states


class ModernBertLMHead(MegatronModule):
    """[修正] 内部のLayerNormを標準版クラス(をラップしたもの)に変更"""
    def __init__(self, mpu_vocab_size, hidden_size, config, parallel_output):
        super().__init__(config=config)
        args = get_args()
        self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
        tensor_parallel.set_tensor_model_parallel_attributes(self.bias, True, 0, 1)
        self.parallel_output = parallel_output
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        # [修正] torch.nn.Linearを使いつつ、sequence_parallel属性をセットする
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        if config.sequence_parallel:
            setattr(self.dense.weight, 'sequence_parallel', True)
            setattr(self.dense.bias, 'sequence_parallel', True)
            print(f"[ModernBertLMHead] sequence_parallel属性をセットしました: {self.dense.weight.sequence_parallel}, {self.dense.bias.sequence_parallel}")
        self.layernorm = BiaslessMixedFusedLayerNorm(hidden_size, eps=config.layernorm_epsilon, sequence_parallel=config.sequence_parallel)
        self.gelu = F.gelu
    def forward(self, hidden_states, word_embeddings_weight):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        output = parallel_lm_logits(hidden_states, word_embeddings_weight, self.parallel_output, bias=self.bias)
        return output


class ModernBertModel(MegatronModule):
    def __init__(self, config, num_tokentypes=2, parallel_output=True):
        super().__init__(config=config)
        args = get_args()
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        
        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(args.padded_vocab_size, config.hidden_size, config=config, init_method=config.init_method)
        self.position_embeddings = torch.nn.Embedding(args.max_position_embeddings, config.hidden_size)
        
        self.embedding_layernorm = BiaslessMixedFusedLayerNorm(config.hidden_size, eps=config.layernorm_epsilon, sequence_parallel=config.sequence_parallel)
        self.embedding_dropout = torch.nn.Dropout(config.hidden_dropout)

        self.transformer_layers = torch.nn.ModuleList(
            [ModernTransformerLayer(config, i, 1) for i in range(config.num_layers)]
        )
        self.lm_head = ModernBertLMHead(args.padded_vocab_size, config.hidden_size, config, parallel_output)
        # self.token_type_embeddings = torch.nn.Embedding(num_tokentypes, config.hidden_size)

    def _get_extended_attention_mask(self, attention_mask):
        attention_mask_b1s = attention_mask.unsqueeze(1)
        attention_mask_bs1 = attention_mask.unsqueeze(2)
        attention_mask_bss = attention_mask_b1s * attention_mask_bs1
        extended_attention_mask = attention_mask_bss.unsqueeze(1)
        return (extended_attention_mask < 0.5)

    def set_input_tensor(self, input_tensor):
        pass

    def forward(self, input_ids, attention_mask, tokentype_ids=None, lm_labels=None):
        # =================================================================
        # >>>>>>>>>>>>>>>>>>>> [デバッグコード] <<<<<<<<<<<<<<<<<<<<
        # =================================================================
        # =======================【拡張デバッグコード】=========================
        # --- word_embeddings の入力チェック ---
        print("--- input_ids check ---")
        print(f"Shape: {input_ids.shape}, Dtype: {input_ids.dtype}")
        print(f"Min: {input_ids.min().item()}, Max: {input_ids.max().item()}")
        print(f"Vocab Size: {self.word_embeddings.weight.shape[0]}")

        word_embeds = self.word_embeddings(input_ids)

        # --- position_embeddings の入力チェック ---
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        print("\n--- position_ids check ---")
        print(f"Shape: {position_ids.shape}, Dtype: {position_ids.dtype}")
        print(f"Min: {position_ids.min().item()}, Max: {position_ids.max().item()}")
        print(f"Max Position Embeddings: {self.position_embeddings.weight.shape[0]}")

        position_embeds = self.position_embeddings(position_ids)

        # --- tokentype_embeddings の入力チェック (既存のコード) ---
        if tokentype_ids is None:
            tokentype_ids = torch.zeros_like(input_ids)
        print("\n--- tokentype_ids check ---")
        print(f"Shape: {tokentype_ids.shape}, Dtype: {tokentype_ids.dtype}")
        # print(f"Min: {tokentype_ids.min().item()}, Max: {tokentype_ids.max().item()}")
        # print(f"Token Type Embedding Size: {self.token_type_embeddings.weight.shape[0]}")

        print("------------------------------------------\n")
        # =================================================================

        if 'mask_validated' not in globals():
            globals()['mask_validated'] = True  # この検証を1回だけ実行するためのフラグ
            
            print("\n" + "="*50)
            print("      Attention Mask の検証（最初の1ステップ）")
            print("="*50)
            
            # 渡されてきたattention_mask（=padding_mask）の形状と中身を確認
            print(f"\nInput attention_mask shape (batch): {attention_mask.shape}")
            print(f"Input attention_mask (sample 0):\n{attention_mask[0].tolist()}")

            # 内部で変換された extended_attention_mask の形状と中身を確認
            extended_attention_mask = self._get_extended_attention_mask(attention_mask)
            print(f"\nExtended attention_mask shape: {extended_attention_mask.shape}")
            print(f"Extended attention_mask (top-left 10x10 slice of sample 0):\n{extended_attention_mask[0, 0, :10, :10].cpu().numpy()}")
            
            # attention_maskで実際にパディングが始まっている位置を確認
            pad_start_index = (attention_mask[0] == 0).nonzero(as_tuple=True)[0]
            if pad_start_index.numel() > 0:
                print(f"\nSample 0 のパディング開始位置: index {pad_start_index[0].item()}")
            else:
                print("\nSample 0 にはパディングがありません。")

            print("\n" + "="*50)
            # import sys; sys.exit("マスク検証のため終了") # 必要に応じてコメントを外してプログラムを止める
        # =================================================================
        # >>>>>>>>>>>>>>>>>>>> デバッグコードここまで <<<<<<<<<<<<<<<<<<<<<<<<<
        # =================================================================
        
        # attention_maskからextended_attention_maskを再生成（デバッグコード内で一度実行済みのため）
        extended_attention_mask = self._get_extended_attention_mask(attention_mask)

        # Embedding層
        word_embeds = self.word_embeddings(input_ids)
        
        # Position Embeddingの追加
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embeddings(position_ids)

        if tokentype_ids is None:
            # デフォルトで全て0（単一セグメント）のテンソルを作成
            tokentype_ids = torch.zeros_like(input_ids)

        # 3つの埋め込みを足し合わせる
        embeddings = word_embeds + position_embeds
        # =======================【追加のデバッグコード】=========================
        print("\n--- embeddings check before layernorm ---")
        print(f"Shape: {embeddings.shape}, Dtype: {embeddings.dtype}")
        # .any() を使って、一つでもNaN/InfがあればTrueと表示させる
        print(f"Contains NaN: {torch.isnan(embeddings).any().item()}")
        print(f"Contains Inf: {torch.isinf(embeddings).any().item()}")
        print("-----------------------------------------\n")

        # =======================【デバッグのための変更】=========================
        # 元のコードをコメントアウト
        # hidden_states = self.embedding_layernorm(embeddings)

        # 一時的にLayerNormをスキップする
        print("!!! DEBUG: Skipping embedding_layernorm !!!")
        hidden_states = embeddings
        # =====================================================================
        # LayerNormとDropout # デバッグ用の遅延（必要に応じて削除）
        hidden_states = self.embedding_layernorm(embeddings)
        hidden_states = self.embedding_dropout(hidden_states)
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        
        
        # Transformer層 (LayerNorm2回問題も解消済み)
        hidden_states = self.transformer_layers[0](hidden_states, extended_attention_mask)
        if len(self.transformer_layers) > 1:
            for layer in self.transformer_layers[1:]:
                hidden_states = layer(hidden_states, extended_attention_mask)
        
        # LM Head
        lm_logits = self.lm_head(hidden_states, self.word_embeddings.weight)

        # Loss計算部分は変更なし (トークンごとのlossを返す)
        if lm_labels is None:
            return lm_logits.transpose(0, 1).contiguous()
        else:
            lm_labels_s_b = lm_labels.transpose(0, 1).contiguous()
            if self.fp16_lm_cross_entropy:
                loss_per_token_s_b = tensor_parallel.vocab_parallel_cross_entropy(lm_logits, lm_labels_s_b)
            else:
                loss_per_token_s_b = tensor_parallel.vocab_parallel_cross_entropy(lm_logits.float(), lm_labels_s_b)
            loss_per_token_b_s = loss_per_token_s_b.transpose(0, 1).contiguous()
            return loss_per_token_b_s

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        return super().state_dict(prefix=prefix, keep_vars=keep_vars)
    
    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)