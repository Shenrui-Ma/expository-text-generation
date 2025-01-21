import datasets
import nltk
import tqdm
import torch
import pickle
import os
import numpy as np
import re
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import load_metric

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 定义路径
ds_name = r"F:\Repos\expository-text-generation\content\wiki_cs"
col_name = "title"
model_save_dir = r"F:\Repos\expository-text-generation\data\datasets\model\retriever_v2"  # 修改为本地保存路径

# 创建保存目录
os.makedirs(model_save_dir, exist_ok=True)

# 加载数据集
data = datasets.load_from_disk(ds_name)  # 本地加载数据集
train_data, test_data = data["train"], data["test"]

num_train, num_test = len(train_data["title"]), len(test_data["title"])

# 计算句子长度
sentence_length = int(
    np.median(
        [
            sum([len(nltk.sent_tokenize(p)) for p in output.split("<paragraph>")])
            for output in train_data["output_aug"]
        ]
    )
)


def convert_data(data):
    ret = {"text": [], "label": []}
    output = data["output_aug"]

    for i in tqdm.tqdm(range(len(data["output_aug"]))):
        out = output[i].split("\n\n")[0]
        out_sentences = nltk.sent_tokenize(out)

        for idx, sent in enumerate(out_sentences):
            ret["text"].append(sent)
            ret["label"].append(idx)

    return datasets.Dataset.from_dict(ret)


# 处理数据
train_proc = convert_data(train_data)
test_proc = convert_data(test_data)

# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


# 对数据进行分词处理
train_tok = train_proc.map(preprocess_function, batched=True)
test_tok = train_proc.map(preprocess_function, batched=True)

# 创建数据整理器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 初始化模型
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=len(np.unique(train_proc["label"]))
)

# 定义训练参数
training_args = TrainingArguments(
    output_dir=os.path.join(model_save_dir, "checkpoints"),  # 检查点保存路径
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    save_strategy="epoch",  # 每个epoch保存一次
    save_total_limit=3,  # 保存最近的3个检查点
)

# 定义评估指标
metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


# 创建训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=test_tok,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# 训练模型
trainer.train()

# 保存模型和分词器
model.save_pretrained(model_save_dir)
tokenizer.save_pretrained(model_save_dir)

# 保存训练配置（可选）
training_config = {
    "num_train_samples": num_train,
    "num_test_samples": num_test,
    "sentence_length": sentence_length,
    "training_args": training_args.to_dict(),
}

with open(os.path.join(model_save_dir, "training_config.pkl"), "wb") as f:
    pickle.dump(training_config, f)

print(f"Model and tokenizer saved to: {os.path.abspath(model_save_dir)}")
print(
    f"Training configuration saved to: {os.path.abspath(os.path.join(model_save_dir, 'training_config.pkl'))}"
)

# 如何加载保存的模型（示例代码）
"""
# 加载模型和分词器
loaded_model = AutoModelForSequenceClassification.from_pretrained(model_save_dir)
loaded_tokenizer = AutoTokenizer.from_pretrained(model_save_dir)

# 加载训练配置（如果需要）
with open(os.path.join(model_save_dir, "training_config.pkl"), "rb") as f:
    loaded_config = pickle.load(f)
"""
