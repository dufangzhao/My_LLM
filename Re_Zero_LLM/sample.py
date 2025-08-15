import os
import torch
from model import GPT, GPTConfig
from transformers import AutoTokenizer

# 配置参数
checkpoint_save_dir = 'checkpoints'
device = 'cuda'
device_type = 'cuda'
dtype = 'bfloat16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# 生成参数
top_k = 50
max_new_tokens = 50

print("加载模型...")

# 加载checkpoint
ckpt_path = os.path.join(checkpoint_save_dir, 'checkpoint.pt')
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

# 初始化模型
args = checkpoint['model_args']
model = GPT(GPTConfig(**args))

# 加载权重
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()
model.to(device)

# 加载tokenizer
tokenizer_path = "tokenizer/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

print("模型加载完成！")


def test_generation(prompt):
    """测试单句生成"""
    print(f"\n输入: {prompt}")

    # 编码
    start_ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    # 生成
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, top_k=top_k)
            generated_text = tokenizer.decode(y[0].tolist())

    print(f"输出: {generated_text}")
    return generated_text


# 测试
if __name__ == "__main__":
    # 单句测试
    test_generation("我们可以做朋友嘛？")

    # 或者交互式测试
    while True:
        user_input = input("\n请输入测试文本 (输入'quit'退出): ")
        if user_input.lower() == 'quit':
            break
        test_generation(user_input)