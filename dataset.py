# -*- coding:utf-8 -*-
import torch
from transformers import BertTokenizerFast
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
from torch.utils.data import Dataset, DataLoader
from utils import load_data
from tqdm import tqdm
import time, json, os


class SentenceDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_path, vocab_path='chinese-roberta-wwm-ext', max_length=128):
        self.data_path = data_path
        self.tokenizer = BertTokenizerFast.from_pretrained(vocab_path)
        self.max_length = max_length
        self.sentence, self.labels, self.length = self.__get_data(data_path)

    def __getitem__(self, index):
        encoding = self.tokenizer.encode_plus(self.sentence[index], truncation=True, padding='max_length',
                                              max_length=self.max_length,
                                              return_tensors="pt")
        item = {key: val.clone().detach().squeeze() for key, val in encoding.items()}  # val的shape为[1,128],要转为128，把1维去掉
        item['labels'] = torch.tensor(int(self.labels[index]))
        return item

    def __len__(self):
        return len(self.labels)

    @staticmethod  # 静态方法，此方法不传入代表实例对象的self参数，并且不强制要求传递任何参数，可以被类直接调用，当然实例化的对象也可以调用。
    def __get_data(dir):
        sentence, label, length = load_data(dir, return_length=True)
        return sentence, label, length

    def get_length_of_single_judgment(self):
        return self.length


class AbstractDataset(Dataset):
    def __init__(self, data_path, vocab_path='chinese-roberta-wwm-ext/vocab.txt', max_length=512):
        word2idx, keep_tokens = load_chinese_base_vocab(vocab_path, simplfied=True)
        self.idx2word = {k: v for v, k in word2idx.items()}
        self.tokenizer = Tokenizer(word2idx)
        self.max_length = max_length
        self.segments, self.summarys = self.__get_data(data_path)

    def __getitem__(self, index):
        ## 得到单个数据
        segment = self.segments[index]
        summary = self.summarys[index]
        token_ids, token_type_ids = self.tokenizer.encode(segment, summary, max_length=self.max_length)
        item = {"token_ids": token_ids, "token_type_ids": token_type_ids, }
        return item

    def __len__(self):
        return len(self.segments)

    @staticmethod  # 静态方法，此方法不传入代表实例对象的self参数，并且不强制要求传递任何参数，可以被类直接调用，当然实例化的对象也可以调用。
    def __get_data(dir):
        Segment = []
        Summary = []
        with open(dir, 'r', encoding="utf8") as f:
            for line in tqdm(f, desc=u'读取数据中'):
                data = json.loads(line)
                summary = data.get('summary')
                segment = data.get('label1')
                Segment.append(segment)
                Summary.append(summary)
        return Segment, Summary


# rm：废弃版的函数，不用管
# def rm_get_dataloader(data_path, vocab_path='chinese-roberta-wwm-ext', max_length=128, batch_size=64, shuffle=False,
#                       save_path=None):
#     """输入data_path和vocab_path：词典和数据文件路径
#     max_length：最大长度
#     batch_size：数据的batch大小
#     shuffle：每个epoch是否打乱
#     save_path：是否保存encoding向量
#     返回dataloader类"""
#     tokenizer = BertTokenizerFast.from_pretrained(vocab_path)
#     sentence, label = load_data(data_path)
#     encoding = tokenizer.batch_encode_plus(sentence, truncation=True, padding=True, max_length=max_length,
#                                            return_tensors="pt")  # {'input_ids': [101, ... 102], 'token_type_ids': [0, ..., 0], 'attention_mask': [1, ..., 1]} ##字的编码;标识是第一个句子还是第二个句子;标识是不是填充
#     if save_path:
#         torch.save(encoding, save_path)
#     dataset = SentenceDataset(encoding, label)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                              shuffle=shuffle)  # shuffle=True:即在每个epoch重新打乱洗牌，epoch内不影响，不同epoch的相同index的batch得到的数据不同
#     return dataloader


if __name__ == '__main__':
    print(os.getcwd())
    train_dataset = AbstractDataset(data_path='data/abstract_data/summary-segment.json')
