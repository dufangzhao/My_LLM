import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.0
    bias: bool = True


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.qkv_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)  # qkv合并为一个大矩阵，方便计算
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)  # output projection
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        # 记得有一篇论文说head_size要等于seq_length才合理
        self.head_size = config.n_embed // config.n_head

    def forward(self, x):
        B, T, C = x.size()  # bach, seq, embed
        q, k, v = self.qkv_attn(x).split(self.n_embed, dim=2)  # qkv大矩阵分为q,k,v3个小矩阵
        q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        k.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        v.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout if self.training else 0,
                                           is_causal=True)  # is_causal为因果掩码，即当前位置之前的位置不能被访问
        # (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)  q * k.T/sqrt(hs)
        # (B, nh, T, T)  * (B, nh, T, hs) -> (B, nh, T, hs)  y = (q * k.T/sqrt(hs)) * v
        y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, nh, hs) -> (B,T,C)
        y = self.resid_dropout(self.c_proj(y))  # (B,T,C)
        return y


class MLP(nn.Module):
    # MLP部分参考llama MLP结构
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed, bias=config.bias)
        self.act = nn.functional.relu
        self.dropout = nn.Dropout(config.dropout)
        self.gate = nn.Linear(config.n_embed, 4 * config.n_embed, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        gate_proj = self.gate(x)
        # llama中的代码：
        # intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
        # nanogpt的
        # x = self.act_func(x)
        # 发现这里区别主要在，nanogpt对upproj的x使用激活函数，llama则是对gate使用
        x = self.act(gate_proj) * x
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln = LayerNorm(config.n_embed, bias=False)  # RMS Norm
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_(x))  # pre-norm
        x = x + self.mlp(self.ln(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embed),  # token embedding
            wpe=nn.Embedding(config.block_size, config.n_embed),  # position embedding
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln=LayerNorm(config.n_embed, bias=False),
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # 共享参数 词表(tokens)->词向量(embedding)/词向量->词表
        num_params = 0
        self.apply(self._init_weights)  # linear 和 embedding 初始化
        for pname, p in self.named_parameters():
            num_params += p.numel()
            if pname.endswith('bias'): # _init_weight中只是先置零，这里精细调整偏置
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        print(f'模型参数量为：{num_params}')

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):  # targets是训练时传入的目标，用来计算交叉熵loss
        device = idx.device
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        # embedding
        tok_emb = self.transformer.wte(idx)  # (B,T,n_embed)
        pos_emb = self.transformer.wpe(pos)  # (T,n_embed)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln(x)
        # 经过lm_head
        # target= True 表示模型正在训练阶段，需要回传loss
        # logits取最后一个（-1）即生成出来的东西，这样和目标的一个token维度相同，才好计算损失
        if targets is not None:  # 训练阶段
            logits = self.lm_head(x) # (B,T,vocab_size)
            # (B*T,vocab_size),(B*T,) 忽略标签-1, 计算交叉熵损失
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:  # 预测阶段
            logits = self.lm_head(x)
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # 对二维参数进行权重衰减
        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for pn, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"{num_decay_params}个参数使用权重衰减，{num_nodecay_params}个参数不使用权重衰减")
        # 启用fused
        fused_avail = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = (device_type == 'cuda') and fused_avail
        if use_fused:
            print("AdamW optimiser using fused!")
        extra_args = {'fused': True} if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        # ** 用于将一个字典解包成关键字参数传递给函数
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # [B,T,vocab_size] 取最后一个生成的token的logits
            # tempreture更高，生成的随机性更高
            # 从这里能知道，是softmax的性质决定的，指数函数小的时候变化小，不同token的probs差距会被减少，随机性就强了
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')  # 忽略topk名以后的token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx