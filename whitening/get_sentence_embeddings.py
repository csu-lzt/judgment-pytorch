'''
第一步，先得到所有的句子向量，分为CLS和AVG两种方法
'''
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from dataset import SentenceDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


class SentenceEmbedding(nn.Module):
    def __init__(self, bert_path):
        super(SentenceEmbedding, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)  # 导入模型超参数
        self.bert = BertModel.from_pretrained(bert_path)  # 加载预训练模型权重

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs[0]
        cls_pooler = outputs[1]  # 池化后的输出 [bs, config.hidden_size]
        avg_pooler = self.AvgPooling(last_hidden_state, attention_mask)
        return avg_pooler

    def AvgPooling(self, input_x, mask):
        input_mask = torch.einsum('blh,bl->blh', input_x, mask)
        temp1 = torch.sum(input_mask, dim=1)
        temp2 = torch.sum(mask, dim=1)
        output = torch.div(torch.sum(input_mask, dim=1).t(), torch.sum(mask, dim=1)).t()  # 除法这里要转置
        return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_path = 'chinese-roberta-wwm-ext'
embedding_layer = SentenceEmbedding(bert_path).to(device)
embedding_layer.eval()

# ================================数据集================================================
batch_size = 8
train_dataset = SentenceDataset(data_path='data/classify_data/train_data.json')
valid_dataset = SentenceDataset(data_path='data/classify_data/valid_data.json')
test_dataset = SentenceDataset(data_path='data/classify_data/test_data.json')
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0,
                          pin_memory=True)  # num_workers=4, pin_memory=True 内存充足时使用，可以加速///1080只能设置num_works=0
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

# ======================================================================================
sentence_embeddings = []
with torch.no_grad():
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        embedding = embedding_layer(input_ids=input_ids, attention_mask=attention_mask).cpu().numpy()
        sentence_embeddings.extend(embedding)

sentence_embeddings = np.array(sentence_embeddings)
print(sentence_embeddings.shape)
save_path = 'whitening/embedding_avg.npy'
np.save(save_path, sentence_embeddings)
