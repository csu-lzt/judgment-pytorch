# -*- coding:utf-8 -*-
import torch, time
from tqdm import tqdm
from bert_seq2seq.utils import load_bert
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
from dataset import AbstractDataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: " + str(device))
vocab_path = 'chinese-roberta-wwm-ext/vocab.txt'
word2idx, keep_tokens = load_chinese_base_vocab(vocab_path, simplfied=True)
model_name = 'roberta'
model_path = 'chinese-roberta-wwm-ext/pytorch_model.bin'
model_save_path = 'model/abs'
data_path = 'data/abstract_data/summary-segment.json'
lr = 1e-5
maxlen = 256
epoch = 5
batch_size = 2


def collate_fn(batch):
    # 动态padding， batch为一部分sample
    def padding(indice, max_length, pad_idx=0):
        # pad 函数
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return token_ids_padded, token_type_ids_padded, target_ids_padded


class ModelTrainer(object):
    def __init__(self, epoch, batch_size):
        # 定义模型
        self.epoch = epoch
        self.model = load_bert(word2idx, model_name=model_name, model_class="seq2seq")  # 加载预训练的模型参数
        self.get_parameter_number()
        self.model.load_pretrain_params(model_path, keep_tokens=keep_tokens)  # 加载已经训练好的模型，继续训练
        self.model.set_device(device)  # 将模型发送到计算设备(GPU或CPU)
        # 声明自定义的数据加载器
        train_dataset = AbstractDataset(data_path=data_path)
        # - shuffle：设置为True的时候，每个epoch都会打乱数据集
        # - collate_fn：如何取样本的，我们可以定义自己的函数来准确地实现想要的功能
        # - drop_last：告诉如何处理数据集长度除于batch_size余下的数据。True就抛弃，否则保留
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.get_parameter_number()

    def get_parameter_number(self):
        # 打印模型参数量
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Total parameters: {total_num}, Trainable parameters: {trainable_num}')

    def model_train(self):
        self.optim_parameters = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)
        # 一个epoch的训练
        self.model.train()
        total_loss = 0
        start_time = time.time()  ## 得到当前时间
        step = 0
        report_loss = 0
        for token_ids, token_type_ids, target_ids in tqdm(self.train_loader):
            step += 1
            if step % 1000 == 0:
                self.model.eval()
                test_data = [
                    "夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及 时 就医 。",
                    "2007年乔布斯向人们展示iPhone并宣称它将会改变世界还有人认为他在夸大其词然而在8年后以iPhone为代表的触屏智能手机已经席卷全球各个角落未来智能手机将会成为真正的个人电脑为人类发展做出更大的贡献",
                    "8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午 ，华 住集 团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。"]
                for text in test_data:
                    print(self.model.generate(text, beam_size=3))
                print("loss is " + str(report_loss))
                report_loss = 0
                # self.eval(epoch)
                self.model.train()
            if step % 8000 == 0:
                self.save(model_save_path)

            # 因为传入了target标签，因此会计算loss并且返回
            predictions, loss = self.model(token_ids,
                                           token_type_ids,
                                           labels=target_ids,
                                           )
            report_loss += loss.item()
            # 反向传播
            # 清空之前的梯度
            self.optimizer.zero_grad()
            # 反向传播, 获取新的梯度
            loss.backward()
            # 用获取的梯度更新模型参数
            self.optimizer.step()

            # 为计算当前epoch的平均loss
            total_loss += loss.item()

        end_time = time.time()
        spend_time = end_time - start_time
        # 打印训练信息
        print("epoch is " + str(epoch) + ". loss is " + str(total_loss) + ". spend time is " + str(spend_time))
        # 保存模型
        self.save(model_save_path)

    def save(self, save_path):
        # 保存模型
        self.model.save_all_params(save_path)
        print("{} saved!".format(save_path))


if __name__ == '__main__':
    trainer = ModelTrainer(epoch=epoch, batch_size=batch_size)
    trainer.model_train()
    # train_epoches = 20
    # for epoch in range(train_epoches):
    #     # 训练一个epoch
    #     trainer.train(epoch)
