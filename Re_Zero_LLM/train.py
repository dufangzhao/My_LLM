import os
import time
import math
import torch
import numpy as np
import torch.nn as nn
from model import GPTConfig,GPT

# 模型参数
block_size = 128  # 窗口大小GPT2为1024
gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
batch_size = 32  # if gradient_accumulation_steps > 1, this is the micro-batch size
n_layer = 12
n_head = 6
n_embed = 768
bias = False
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
init_from = 'scratch'  # 'scratch' or 'resume' # 从头训练还是继续
checkpoint_save_dir = 'checkpoints'
eval_iters = 200
eval_interval = 200  # 每n步eval和保存checkpoint一次
always_save_checkpoint = True

# 学习率衰减 - 调整为更合理的值
learning_rate = 3e-4  # 降低学习率
warmup_iters = 200
lr_decay_iters = 2000  # 延长衰减周期
min_lr = 3e-5

# 优化器参数
max_iters = 2000  # 增加训练步数
weight_decay = 1e-1
betas = (0.9, 0.95)
grad_clip = 1.0  # 梯度裁剪

# system
device = 'cuda'
device_type = 'cuda'
compile = True
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
# 在后续上下文中会按照dtype以及具体运算方法自动调整精度 dtype参数类似于基准精度
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# dataloader
dataset_path = 'data/vtuber'
data_dir = os.path.join(dataset_path)


def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint32, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint32, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,))  # (batch_size,) 随机取batch_size个样本一维索引矩阵
    x = torch.stack(
        [torch.from_numpy((data[i:i + block_size].astype(np.int64))) for i in ix])  # (batch_size, block_size)
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size].astype(np.int64))) for i in
                     ix])  # (batch_size, block_size) x向后移一位，作为训练时的预测标签
    # 固定在内存中，直接从内存访问，并不经过cpu，non_blocking表示异步操作，不影响cpu继续运行代码
    print(x.device)
    print(y.device)
    x, y = x.contiguous().pin_memory().to(device, non_blocking=True), y.contiguous().pin_memory().to(device,
                                                                                                     non_blocking=True)
    return x, y


model_args = dict(n_layer=n_layer, n_head=n_head, n_embed=n_embed, block_size=block_size, bias=bias, vocab_size=None,
                  dropout=dropout)

iter_num = 0
best_val_loss = 1e9

assert init_from == 'scratch' or init_from == 'resume'
if init_from == 'scratch':
    print("从头训练模型")
    # 根据prepare.py的输出，最大token ID是151603，所以设置为151604
    model_args['vocab_size'] = 129184  # 修正词汇表大小
    gpt_args = GPTConfig(**model_args)
    model = GPT(gpt_args)

elif init_from == 'resume':
    print("继续训练模型")
    ckpt_path = os.path.join(checkpoint_save_dir, 'checkpoint.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embed', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gpt_args = GPTConfig(**model_args)
    model = GPT(gpt_args)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))  # 当目前混合精度为float16时启动梯度自动缩放，防止梯度下溢
model.to(device)
optimizer = model.configure_optimizers(weight_decay, learning_rate, betas, device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None
# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0


# 评估训练集和验证集损失
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(now_iter):
    if (now_iter < warmup_iters):  # lr warmup
        return learning_rate * (now_iter + 1) / (warmup_iters + 1)
    elif (now_iter > lr_decay_iters):  # lr decay
        return min_lr
    else:  # lr cosine decay
        rate = (now_iter - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * rate))  # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)  # ranges min_lr..learning_rate 余弦下降


# 创建checkpoint目录
os.makedirs(checkpoint_save_dir, exist_ok=True)

# 训练
t_before = time.time()
X, Y = get_batch('train')
# 初始评估
if iter_num == 0:
    loss_dict = estimate_loss()
    print(f"初始状态 - train_loss: {loss_dict['train']:.4f}, val_loss: {loss_dict['val']:.4f}")
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num > 0 and iter_num % eval_interval == 0:
        loss_dict = estimate_loss()
        print(f"iter {iter_num} - train_loss: {loss_dict['train']:.4f}, val_loss: {loss_dict['val']:.4f}, lr: {lr:.2e}")
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss
        }
        checkpoint_path = os.path.join(checkpoint_save_dir, f'checkpoint_{iter_num}.pt')
        if loss_dict['val'] < best_val_loss:
            best_val_loss = loss_dict['val']
            print(print(f"新的最佳val loss: {best_val_loss:.4f}"))
            print(f"saving checkpoint to {checkpoint_save_dir}")
            torch.save(checkpoint, checkpoint_path)
        if always_save_checkpoint:
            print(f"saving checkpoint to {checkpoint_save_dir}")
            torch.save(checkpoint, checkpoint_path)
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        X, Y = get_batch('train')
        with ctx:
            logits, loss = model(X, Y)
            # backward pass, with gradient scaling if training in fp16
            loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation
        scaler.scale(loss).backward()
    if iter_num % 50 == 0:  # 每50步打印一次，减少输出
        print(f"iter: {iter_num}, loss: {loss.item():.4f}, lr: {lr:.2e}")
    # clip the gradient
    if grad_clip > 0.0:
        scaler.unscale_(optimizer)  # 损失使用了缩放，梯度反缩放才能进行裁剪
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)  # 使用缩放的梯度执行优化器步骤
    scaler.update()  # 根据梯度的溢出情况动态调整缩放比例
    optimizer.zero_grad(set_to_none=True)  # 清空梯度缓存，为下一次迭代做准

    t_after = time.time()
    dt = t_after - t_before
    t_before = t_after

    iter_num += 1
    if iter_num > max_iters:
        print(f"训练完成！总共训练了{max_iters}步")
        break
# 最终评估
print("进行最终评估...")
final_losses = estimate_loss()
print(f"最终结果 - train_loss: {final_losses['train']:.4f}, val_loss: {final_losses['val']:.4f}")