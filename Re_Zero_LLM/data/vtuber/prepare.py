import os
import numpy as np
from transformers import AutoTokenizer

# 使用本地Qwen2.5 tokenizer
tokenizer_path = "../../tokenizer/Qwen2.5-7B-Instruct"
print(f"正在加载本地tokenizer: {tokenizer_path}")
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path,
    local_files_only=True
)
print("本地tokenizer加载成功！")

# 读取数据
# input_file_path = os.path.join(os.path.dirname(__file__), 'test.txt')
input_file_path = "./test.txt"
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read() # f.read()会自动将utf-8字节序列转换为Unicode字符串
print(f"原始数据长度: {len(data)} 字符")

# 数据分割
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# 使用Qwen2.5 tokenizer编码
print("正在编码训练数据...")
train_ids = tokenizer.encode(train_data, add_special_tokens=False)
print("正在编码验证数据...")
val_ids = tokenizer.encode(val_data, add_special_tokens=False)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
print(f"训练数据压缩比: {len(train_data)/len(train_ids):.2f} 字符/token")
print(f"验证数据压缩比: {len(val_data)/len(val_ids):.2f} 字符/token")

# 检查token范围
max_token_id = max(max(train_ids), max(val_ids))
print(f"最大token ID: {max_token_id}")

# 使用uint32而不是uint16来避免溢出
print("保存为uint32格式...")
train_ids = np.array(train_ids, dtype=np.uint32)  # 改为uint32
val_ids = np.array(val_ids, dtype=np.uint32)     # 改为uint32

# train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
# val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
train_ids.tofile('./train.bin')
val_ids.tofile('./val.bin')
print("数据处理完成！")
print(f"训练数据保存到: train.bin ({len(train_ids)} tokens)")
print(f"验证数据保存到: val.bin ({len(val_ids)} tokens)")