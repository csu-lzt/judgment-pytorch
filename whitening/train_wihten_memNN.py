#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
纯分类模型的训练和测试
去掉memory模块，直接在embedding上变动，这里单纯训练线性分类器
'''
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AdamW, get_cosine_schedule_with_warmup
import time
from utils import load_data
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report


class EmbeddingsDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, embedding_path, data_path):
        self.data = np.load(embedding_path)
        _, self.labels, self.length = load_data(data_path, return_length=True)

    def __getitem__(self, index):
        item = {}
        item['embeddings'] = self.data[index]
        item['labels'] = self.labels[index]
        return item

    def __len__(self):
        return len(self.data)

    def get_length_of_single_judgment(self):
        return self.length


class SentenceClassifyModel(nn.Module):
    def __init__(self, hidden_size=768, classes=2):
        super(SentenceClassifyModel, self).__init__()
        self.classifier = nn.Linear(hidden_size, classes)  # 直接分类

    def forward(self, embeddings):
        logit = self.classifier(embeddings)  # [bs, classes]
        return logit


class ModelTrainer(object):
    def __init__(self, model, epochs, batch_size):
        self.model = model
        self.epochs = epochs
        self.best_acc = 0.0
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                       pin_memory=True)  # num_workers=4, pin_memory=True 内存充足时使用，可以加速///1080只能设置num_works=0
        self.steps = len(self.train_loader)
        self.get_parameter_number()

    def get_parameter_number(self):
        # 打印模型参数量
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Total parameters: {total_num}, Trainable parameters: {trainable_num}')

    def model_train(self, retrain=False):
        # 模型训练
        optimizer = AdamW(self.model.parameters(), lr=2e-5, weight_decay=1e-4)  # AdamW优化器
        ## 学习率先线性warmup一个epoch，然后cosine式下降。这里给个小提示，一定要加warmup（学习率从0慢慢升上去），要不然你把它去掉试试，基本上收敛不了。
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.steps,
                                                    num_training_steps=(self.epochs) * (self.steps))

        for i in range(self.epochs):
            start_time = time.time()
            self.model.train()
            epoch_index = i + 1
            print(f"***** Running training epoch {epoch_index} *****")
            train_loss_sum = 0.0
            train_accuracy = 0.0
            batch_index = 0
            for batch in self.train_loader:
                batch_index += 1
                embeddings = batch['embeddings'].to(device)
                true_labels = batch['labels'].to(device)
                pred_labels = self.model(embeddings)
                loss = nn.CrossEntropyLoss()(pred_labels, true_labels)
                optimizer.zero_grad()  # 先将梯度归零,如果不将梯度清零的话，梯度会与上一个batch的数据相关，因此该函数要写在反向传播和梯度下降之前
                loss.backward()  # 然后反向传播计算得到每个参数的梯度值
                optimizer.step()  # 最后通过梯度下降执行一步参数更新
                scheduler.step()  # 学习率变化
                # 损失和准确率的计算
                train_loss_sum += loss.item()
                train_accuracy += self.model_evaluate(true_labels, pred_labels)
                # if (batch_index) % (self.steps // 50) == 0:  # 只打印五次结果
                print('\r',
                      "Epoch {:04d} | Step {:04d}/{:04d} | Train Cost Time {:.1f}s | End Time {:.1f}s | Step Time {:.1f}s | Loss {:.4f} | Accuracy {:.4f} | LR {}"
                      .format(epoch_index, batch_index, self.steps, time.time() - start_time,
                              (time.time() - start_time) / batch_index * (self.steps - batch_index),
                              (time.time() - start_time) / batch_index,
                              train_loss_sum / batch_index, train_accuracy / batch_index,
                              optimizer.state_dict()['param_groups'][0]['lr']),
                      end='', flush=True)
                # print('\n', "Learning rate = {}".format(optimizer.state_dict()['param_groups'][0]['lr']), end='',
                #       flush=True)
            print('\n')
            # 验证
            # acc = self.model_valid(data_loader=self.valid_loader, desc='Validing.......')
            # if acc > self.best_acc:
            #     self.best_acc = acc
            #     torch.save({'model_state_dict': self.model.state_dict(),
            #                 'epoch': epoch_index,
            #                 'optimizer_state_dict': optimizer.state_dict(),
            #                 }, self.best_model_path)
            # torch.save({'model_state_dict': self.model.state_dict(),
            #             'epoch': epoch_index,
            #             'optimizer_state_dict': optimizer.state_dict(),
            #             }, self.most_model_path)
            # print("\n current val_acc is {:.4f}, best val_acc is {:.4f}".format(acc, self.best_acc))
            # print("Train and Valid Time {:.1f}s \n".format(time.time() - start_time))

    def model_valid(self, data_loader, desc):
        # 返回准确率   #以及真实标签和预测标签
        self.model.eval()
        true_labels, pred_labels = [], []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=desc):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                y_true = batch['labels'].to(device)
                y_pred = self.model(input_ids=input_ids, attention_mask=attention_mask, mode='valid')
                true_labels.extend(y_true.cpu().numpy())
                pred_labels.extend(torch.argmax(y_pred, dim=1).detach().cpu().numpy())
        return accuracy_score(true_labels, pred_labels)

    def model_evaluate(self, y_true, y_pred):
        # 评测函数，计算准确率
        # 转到cpu上，numpy
        np_y_true = y_true.cpu().numpy()
        np_y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy()
        return accuracy_score(np_y_true, np_y_pred)

    def model_test(self):
        checkpoint = torch.load(self.best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # 注意加载模型使用 load_state_dict 方法，其参数不是文件路径，而是 torch.load(PATH)
        self.model.eval()
        true_labels, pred_labels = [], []
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Testing.......'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                y_true = batch['labels'].to(device)
                y_pred = self.model(input_ids=input_ids, attention_mask=attention_mask, mode='test')
                true_labels.extend(y_true.cpu().numpy())
                pred_labels.extend(torch.argmax(y_pred, dim=1).detach().cpu().numpy())
        print("\n Test Accuracy = {} \n".format(accuracy_score(true_labels, pred_labels)))
        print(classification_report(true_labels, pred_labels, digits=2))


# ===============实验设置=====================
# 修改embedding_path和batch_size
epochs = 20
batch_size = 8
print("batch_size", batch_size)
train_dataset = EmbeddingsDataset(embedding_path='whitening/embedding_avg_whiten_memNN.npy',
                                  data_path='data/classify_data/train_data.json')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SentenceClassifyModel(hidden_size=1536).to(device)
model_train = ModelTrainer(model=model, epochs=epochs, batch_size=batch_size)
model_train.model_train()
