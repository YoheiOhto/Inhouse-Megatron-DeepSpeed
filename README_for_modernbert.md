# MODERN BERT in deepspeed-megatron
## argのみで変更できること
- mask率=30% --mask-prob 0.3
- MLMのみでの実行 --bert-no-binary-head をつける
- preLN --apply-residual-connection-post-layernorm をつけない

## モデルの初期化関数  
- [ModernBERTでは切断正規分布](https://github.com/AnswerDotAI/ModernBERT/blob/main/src/bert_layers/initialization.py) full_megatronが選択されている  
  
### 実装方法  
- megatron/model/initialization.py に実装コード
    - 各層にmoduleを付与
    - "--full-megarton-model-init" model構築後に再度weightを与えなおす  
    - 現在の実装では megatron/model/bert_model.py の内部のみに適用するようにしている
    - また、+-2σは実装内で固定している

## モデルの構成 (確認)
```python
    config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
    model = AutoModelForMaskedLM.from_config(config)
    print(model)

    ModernBertForMaskedLM(
    (model): ModernBertModel(
        (embeddings): ModernBertEmbeddings(
        (tok_embeddings): Embedding(50368, 768, padding_idx=50283)
        (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (drop): Dropout(p=0.0, inplace=False)
        )
        (layers): ModuleList(
        (0): ModernBertEncoderLayer(
            (attn_norm): Identity()
            (attn): ModernBertAttention(
            (Wqkv): Linear(in_features=768, out_features=2304, bias=False)
            (rotary_emb): ModernBertRotaryEmbedding()
            (Wo): Linear(in_features=768, out_features=768, bias=False)
            (out_drop): Identity()
            )
            (mlp_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): ModernBertMLP(
            (Wi): Linear(in_features=768, out_features=2304, bias=False)
            (act): GELUActivation()
            (drop): Dropout(p=0.0, inplace=False)
            (Wo): Linear(in_features=1152, out_features=768, bias=False)
            )
        )
        (1-21): 21 x ModernBertEncoderLayer(
            (attn_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): ModernBertAttention(
            (Wqkv): Linear(in_features=768, out_features=2304, bias=False)
            (rotary_emb): ModernBertRotaryEmbedding()
            (Wo): Linear(in_features=768, out_features=768, bias=False)
            (out_drop): Identity()
            )
            (mlp_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): ModernBertMLP(
            (Wi): Linear(in_features=768, out_features=2304, bias=False)
            (act): GELUActivation()
            (drop): Dropout(p=0.0, inplace=False)
            (Wo): Linear(in_features=1152, out_features=768, bias=False)
            )
        )
        )
        (final_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
    (head): ModernBertPredictionHead(
        (dense): Linear(in_features=768, out_features=768, bias=False)
        (act): GELUActivation()
        (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
    (decoder): Linear(in_features=768, out_features=50368, bias=True)
    )
```
```python
    print(model) # debug用にpretrain関数の内部で実行

    [DeepSpeedEngine(
    (module): BertModel(
        (language_model): TransformerLanguageModel(
        (embedding): Embedding(
            (word_embeddings): VocabParallelEmbedding()
            (embedding_dropout): Dropout(p=0.1, inplace=False)
        )
        (embedding_layernorm): MixedFusedLayerNorm()
        (encoder): ParallelTransformer(
            (layers): ModuleList(
            (0-21): 22 x ParallelTransformerLayer(
                (input_layernorm): MixedFusedLayerNorm()
                (self_attention): ParallelAttention(
                (rotary_pos_emb): RotaryEmbedding()
                (query_key_value): ColumnParallelLinear()
                (core_attention): CoreAttention(
                    (scale_mask_softmax): FusedScaleMaskSoftmax()
                    (attention_dropout): Dropout(p=0.1, inplace=False)
                )
                (dense): RowParallelLinear()
                )
                (post_attention_layernorm): MixedFusedLayerNorm()
                (mlp): ParallelMLP(
                (dense_h_to_4h): ColumnParallelLinear()
                (dense_4h_to_h): RowParallelLinear()
                )
            )
            )
            (final_layernorm): MixedFusedLayerNorm()
        )
        )
        (lm_head): BertLMHead(
        (dense): Linear(in_features=768, out_features=768, bias=True)
        (layernorm): MixedFusedLayerNorm()
        )
    )
    )]
```
***(decoder): Linear(in_features=768, out_features=50368, bias=True)***の有無に実装の差がでる 
```python
def get_linear_layer(rows, columns, init_method, gather_params_on_init=False):
    """Simple linear layer with weight initialization."""
    args = get_args()
    layer = torch.nn.Linear(rows, columns, bias=args.add_bias_linear)
    if get_args().perform_initialization:
        with GatheredParameters(layer.weight, modifier_rank=0, enabled=gather_params_on_init):
            init_method(layer.weight)
    if args.add_bias_linear:
        with torch.no_grad():
            with GatheredParameters(layer.bias, modifier_rank=0, enabled=gather_params_on_init):
                layer.bias.zero_()
    return layer
```

  

    
## バイアス項の無効化
| 対象モジュール | バイアスの有無 | 理由 |
|-----|-----|-----|
| Attention & MLP内の全線形層 | なし (bias=False) | 論文の指示 |
| 全てのLayerNorm層 | なし (bias=False) | 論文の指示 |
| BertLMHeadの最終出力 | あり (bias=True) | 論文で唯一の例外 |

### 線形層における実装
* --disable-bias-linear の設定で実装可能

### LayerNorm層における実装
```python
# from .fused_layer_norm import MixedFusedLayerNorm as LayerNorm
if self.train_bias: # default train_bias == False
    self.bias = Parameter(torch.empty(*normalized_shape,
                                        device=init_device,
                                        dtype=get_args().params_dtype))
else:
    print("WARNING: FusedLayerNorm is created without bias parameter.")
    self.register_buffer('bias', torch.zeros(*normalized_shape,
                                                device=init_device,
                                                dtype=get_args().params_dtype))
```

### BertLMHeadの最終出力
```python
self.layernorm = LayerNorm(hidden_size,
                            eps=config.layernorm_epsilon,
                            sequence_parallel=config.sequence_parallel,
                            train_bias=True)
```

## GeGLUの実装
megatron/model/transformer.py  
ここでgegluを定義して呼び出し  
```python
elif args.geglu:
    def geglu(x):
        x = torch.chunk(x, 2, dim=-1)
        return F.gelu(x[0]) * x[1]
    self.activation_func = geglu
    print("Using GEGLU activation function")
```
arguments.pyにも定義  
```python
def core_transformer_config_from_args(args):
    if args.geglu:
        kw_args['activation_func'] = F.gelu
        kw_args['gated_linear_unit'] = True
        kw_args['bias_gelu_fusion'] = False
```

## grobal - local attention
megatron/model/transformer.py class ParallelTransformerLayer(MegatronModule): 
```python
is_global_attention = True
if args.use_switch_attention:
    if layer_number % args.global_attn_every_n_layers != 0:
        is_global_attention = False
print(f"Layer {self.layer_number}: "
        f"use_switch_attention={args.use_switch_attention}, "
        f"is_global={is_global_attention}")
```

## local attention - sliding window
megatron/model/transformer.py class ParallelAttention(MegatronModule)  
ParallelTransformerLayerの内部で呼び出されている  
```python
if not self.is_global_attention:

    seq_len_q = hidden_states.size(0)
    seq_len_k = hidden_states.size(0) 
    window_size = self.local_window_size
    q_indices = torch.arange(seq_len_q, device=hidden_states.device).view(-1, 1)
    k_indices = torch.arange(seq_len_k, device=hidden_states.device).view(1, -1)
    
    relative_indices = k_indices - q_indices
    local_mask = (relative_indices >= -window_size) & (relative_indices <= window_size)
    attention_mask = attention_mask & local_mask.unsqueeze(0).unsqueeze(0)
```  


## global - local rope
megatron/model/transformer.py class ParallelAttention(MegatronModule)  
ParallelTransformerLayerの内部で呼び出されている    
```python
if self.use_switch_attention_rope:
    if self.is_global_attention:
        theta = args.global_rope_theta
    else:
        theta = args.local_rope_theta
    rotary_dim = config.kv_channels
    if args.rotary_percent < 1.0:
        rotary_dim = int(rotary_dim * args.rotary_percent)
    self.rotary_pos_emb = RotaryEmbedding(rotary_dim, theta=theta)
```
*transformer_impl == 'local'* 以外を設定しない!

## optimizerの実装
https://github.com/warner-benjamin/optimi  
このstable adam Wを使用する  
megatron/optimizer/__init__.py
```python
elif args.optimizer == 'stable_adamw':
    optimi_param_groups = []
    for group in param_groups:
        new_group = {
            'params': group['params'],
            'weight_decay': args.weight_decay * group.get('wd_mult', 1.0)
        }
        if group.get('lr_mult', 1.0) != 1.0:
            new_group['lr'] = args.lr * group.get('lr_mult')
        optimi_param_groups.append(new_group)
    optimizer = StableAdamW(optimi_param_groups,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(args.adam_beta1, args.adam_beta2),
                            eps=args.adam_eps,
                            decouple_lr=args.stable_adamw_decouple_lr,
                            max_lr=args.lr if args.stable_adamw_decouple_lr else None,
                            kahan_sum=args.stable_adamw_kahan_sum,
                            triton=False,
                            foreach=True
                            )
```

  
