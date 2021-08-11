# -*- coding:utf-8 -*-
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig, AdamW, get_cosine_schedule_with_warmup
from dataset import SentenceDataset
import numpy as np

bert_path = 'chinese-roberta-wwm-ext'  # 该文件夹下存放三个文件（'vocab.txt', 'pytorch_model.bin', 'config.json'）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### 1. 这种方式是不需要手动下载模型文件，在网速快的时候使用
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert = BertModel.from_pretrained('bert-base-uncased')
### 2. 这种方式是需要手动下载模型文件，网速慢使用，推荐使用第2种。
# tokenizer = BertTokenizer.from_pretrained('E:/Projects/bert-base-uncased/bert-base-uncased-vocab.txt')
# 加载bert模型，这个路径文件夹下有bert_config.json配置文件和model.bin模型权重文件
# bert = BertModel.from_pretrained('chinese-roberta-wwm-ext')

class SentenceClassifyModel(nn.Module):
    def __init__(self, bert_path, classes=2):
        super(SentenceClassifyModel, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)  # 导入模型超参数
        self.bert = BertModel.from_pretrained(bert_path)  # 加载预训练模型权重
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, classes)  # 直接分类

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out_pool = outputs[1]  # 池化后的输出 [bs, config.hidden_size]
        logit = self.classifier(out_pool)  # [bs, classes]
        return out_pool


batch_size = 64
train_dataset = SentenceDataset(data_path='data/classify_data/train_data.json')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                          pin_memory=True)
model = SentenceClassifyModel(bert_path).to(device)
model.load_state_dict(torch.load('model/best_model.pth'))
# 注意加载模型使用 load_state_dict 方法，其参数不是文件路径，而是 torch.load(PATH)
model.eval()
sentence_embeddings = []
with torch.no_grad():
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        y_true = batch['labels'].to(device)
        embedding = model(input_ids=input_ids, attention_mask=attention_mask).cpu().numpy()
        sentence_embeddings.extend(embedding)

sentence_embeddings = np.array(sentence_embeddings)
print(sentence_embeddings.shape)
save_path = 'whitening/finetuned_embedding_cls.npy'
np.save(save_path, sentence_embeddings)
