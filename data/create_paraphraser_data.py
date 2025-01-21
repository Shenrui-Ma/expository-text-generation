import torch
print(torch.cuda.is_available())  # 如果返回 True，说明支持 CUDA

retriever_model_dir ='c:\\Users\\Administrator\\Desktop\\expository-text-generation-main\\model\\retriever'
dataset_dir = 'C:\\Users\\Administrator\\Desktop\\expository-text-generation-main\\wiki_cs'
input_column_name = 'web_sentences_with_desc'
output_dir = 'paraphrase_college_with_desc_title1'
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import datasets
import numpy as np
import tqdm
import torch
import nltk
#data = datasets.load_dataset(dataset_dir)
data = datasets.load_from_disk(dataset_dir)
print(data['train'].column_names)
print(data['test'].column_names)
train_data, test_data,val_data = data['train'], data['test'],data['val']

num_train, num_test ,num_val = len(train_data['title']), len(test_data['title']),len(val_data['title'])

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
model = AutoModelForSequenceClassification.from_pretrained(retriever_model_dir, num_labels=max([len(nltk.sent_tokenize(sent)) for sent in train_data['output_aug']])).to('cpu')

def get_embedding_from_class(sent):
    tok = tokenizer(sent, return_tensors='pt', truncation=True).input_ids.to('cpu')
    return model(tok, output_hidden_states=True).hidden_states[-1].mean(axis = 1).to('cpu').detach()

paraphrase_ds_val = {'title': [], 'facts': [], 'style': [], 'output': []}

all_output_texts = []
all_outputs = val_data['output_aug']
for text in all_outputs:
    all_output_texts.extend(nltk.sent_tokenize(text))
all_output_embs = torch.concat([get_embedding_from_class(s) for s in all_output_texts])

print(len(val_data[input_column_name]))

for data_num in tqdm.tqdm(range(num_val)):
    print(data_num)
    input_text = list(set(val_data[input_column_name][data_num]))
    info_embeddings = []
    for s in input_text:
        info_embeddings.append(get_embedding_from_class(s))
    info_embeddings = torch.concat(info_embeddings)

    for sent in nltk.sent_tokenize(val_data['output'][data_num]):
        sent_emb = get_embedding_from_class(sent)
        style_sims = (all_output_embs @ sent_emb.T).squeeze(1)
        for idx in np.random.choice(torch.argsort(style_sims, descending=True)[1:11], 10):
            curr_style = all_output_texts[idx]
            style_emb = get_embedding_from_class(sent)
            fact_sims = (info_embeddings @ sent_emb.T).squeeze(1)
            
            facts = [input_text[idx_] for idx_ in torch.argsort(fact_sims, descending=True)[:10]]

            paraphrase_ds_val['facts'].append(facts)
            paraphrase_ds_val['style'].append(curr_style)
            paraphrase_ds_val['output'].append(sent)
            paraphrase_ds_val['title'].append(val_data['title'][data_num])


paraphrase_ds_train = {'title': [], 'facts': [], 'style': [], 'output': []}

all_output_texts = []
all_outputs = train_data['output_aug']
for text in all_outputs:
    all_output_texts.extend(nltk.sent_tokenize(text))
all_output_embs = torch.concat([get_embedding_from_class(s) for s in all_output_texts])

for data_num in tqdm.tqdm(range(num_train)):
    break
    input_text = list(set(train_data[input_column_name][data_num]))
    info_embeddings = []
    for s in input_text:
        info_embeddings.append(get_embedding_from_class(s))
    info_embeddings = torch.concat(info_embeddings)

    for sent in nltk.sent_tokenize(train_data['output_aug'][data_num]):
        sent_emb = get_embedding_from_class(sent)
        style_sims = (all_output_embs @ sent_emb.T).squeeze(1)
        for idx in np.random.choice(torch.argsort(style_sims, descending=True)[1:50], 10):
            curr_style = all_output_texts[idx]
            style_emb = get_embedding_from_class(sent)
            fact_sims = (info_embeddings @ sent_emb.T).squeeze(1)
            
            facts = [input_text[idx_] for idx_ in torch.argsort(fact_sims, descending=True)[:10]]

            paraphrase_ds_train['facts'].append(facts)
            paraphrase_ds_train['style'].append(curr_style)
            paraphrase_ds_train['output'].append(sent)
            paraphrase_ds_train['title'].append(train_data['title'][data_num])



paraphrase_ds_test = {'title': [], 'facts': [], 'style': [], 'output': []}

all_output_texts = []
all_outputs = val_data['output_aug']
for text in all_outputs:
    all_output_texts.extend(nltk.sent_tokenize(text))
all_output_embs = torch.concat([get_embedding_from_class(s) for s in all_output_texts])

for data_num in tqdm.tqdm(range(num_test)):
    break
    input_text = list(set(test_data[input_column_name][data_num]))
    info_embeddings = []
    for s in input_text:
        info_embeddings.append(get_embedding_from_class(s))
    info_embeddings = torch.concat(info_embeddings)

    for sent in nltk.sent_tokenize(test_data['output'][data_num]):
        sent_emb = get_embedding_from_class(sent)
        style_sims = (all_output_embs @ sent_emb.T).squeeze(1)
        for idx in np.random.choice(torch.argsort(style_sims, descending=True)[1:11], 10):
            curr_style = all_output_texts[idx]
            style_emb = get_embedding_from_class(sent)
            fact_sims = (info_embeddings @ sent_emb.T).squeeze(1)
            
            facts = [input_text[idx_] for idx_ in torch.argsort(fact_sims, descending=True)[:10]]

            paraphrase_ds_test['facts'].append(facts)
            paraphrase_ds_test['style'].append(curr_style)
            paraphrase_ds_test['output'].append(sent)
            paraphrase_ds_test['title'].append(test_data['title'][data_num])

new_train, new_test ,new_val= datasets.Dataset.from_dict(paraphrase_ds_train), datasets.Dataset.from_dict(paraphrase_ds_test),datasets.Dataset.from_dict(paraphrase_ds_val)
new_ds = datasets.DatasetDict({'train': new_train, 'test': new_test, 'val': new_val})

new_ds.save_to_disk(output_dir)
    