import os
import pandas as pd
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling,
)
from huggingface_hub import HfApi, HfFolder
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一个 GPU
# 设置数据集路径和模型保存路径
ds_name = r"E:\GitHub\Documents\IRP\expository-text-generation\data\datasets\wiki_cs\content\wiki_cs_new"
final_name = "giovannaadam/imitator_model3"
hf_token = "HF_TOKEN"

# 加载数据集
data = load_from_disk(ds_name)
print(data.keys())
history_text = [h.split("\n\n")[0] for h in data["train"]["output_aug"]]

# 创建输出目录
output_dir = r"E:\GitHub\Documents\IRP\expository-text-generation\model\imitator_model3"
hf_token = "HF_TOKEN"
os.makedirs(output_dir, exist_ok=True)

# 保存处理后的数据
data_file = os.path.join(output_dir, "data.txt")
with open(data_file, "w", encoding="utf8") as f:
    for line in history_text:
        f.write(f"{line}\n")

# 加载模型和分词器
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 创建数据集
train_dataset = TextDataset(tokenizer=tokenizer, file_path=data_file, block_size=128)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=1000,
    save_total_limit=2,
    logging_dir="./logs",
)

# 训练模型
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()

# 保存模型
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# 上传模型到 Hugging Face
HfFolder.save_token(hf_token)
api = HfApi()
api.create_repo(final_name, repo_type="model")
api.upload_folder(folder_path=output_dir, repo_id=final_name, repo_type="model")
