import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from sklearn.metrics import roc_auc_score

# 定义 Transformer 模型
class TransformerACP(nn.Module):
    def __init__(self, input_channels, sequence_length, nhead=4, num_layers=2):
        super(TransformerACP, self).__init__()
        self.embedding = nn.Linear(input_channels, 64)
        self.positional_encoding = self._generate_positional_encoding(sequence_length, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(64 * sequence_length, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)





        # self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()






    def _generate_positional_encoding(self, seq_len, d_model):
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        """
        前向传播函数。

        参数:
        x (torch.Tensor): 输入数据。

        返回:
        torch.Tensor: 模型输出。
        """
        x = self.embedding(x.transpose(1, 2))
        x = x + self.positional_encoding[:, :x.size(1), :].to(x.device)
        x = self.transformer_encoder(x.transpose(0, 1))
        x = x.transpose(0, 1).contiguous().view(x.size(1), -1)
        x = self.relu(self.fc1(x))




        # x = self.dropout(self.relu(self.fc1(x)))
        # x = self.fc2(x)
        x = self.sigmoid(self.fc2(x))





        return x

# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs, Logging):
    all_losses = []
    all_accuracies = []
    for epoch in range(num_epochs):
        TimeTransformerNum = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()




            # probs = torch.sigmoid(outputs.squeeze())
            # predicted = (probs > 0.3).int()
            predicted = (outputs.squeeze() > 0.5).int()





            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        all_losses.append(epoch_loss)
        all_accuracies.append(epoch_accuracy)
        Logging.info(f'训练次数: {epoch + 1}/{num_epochs}, 丢失率: {epoch_loss:.4f}, 准确率: {epoch_accuracy:.2f}%, 耗时: {time.time() - TimeTransformerNum:.4f} 秒')

    return all_losses, all_accuracies

# 主函数
def transformer_train(paraDict, all_sequences, all_labels, Logging, TimeStart):
    try:
        Logging.info("正常：准备运行'transformer'模型")

        # 创建数据集和数据加载器
        dataset = TensorDataset(all_sequences, all_labels)
        train_loader = DataLoader(dataset, batch_size=int(paraDict['BatchSize']), shuffle=True)

        # 模型实例化
        input_channels = all_sequences.size(1)
        sequence_length = all_sequences.size(2)
        model = TransformerACP(input_channels, sequence_length)

        # 定义损失函数和优化器





        # neg_count = (all_labels == 0).sum().item()
        # pos_count = (all_labels == 1).sum().item()
        # pos_weight = torch.tensor([neg_count / pos_count]).to(all_labels.device)
        # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        criterion = nn.BCELoss()
        # optimizer = optim.Adam(model.parameters(), lr=1e-4)
        optimizer = optim.Adam(model.parameters(), lr=float(paraDict['LearningRate']))






        # 训练模型
        num_epochs = int(paraDict['Epochs'])
        TimeTransformer_T = time.time()
        all_losses, all_accuracies = train_model(model, train_loader, criterion, optimizer, num_epochs, Logging)
        Logging.info(f"总训练耗时: {time.time() - TimeTransformer_T:.4f} 秒")

        # 保存模型
        model_save_path = "backend/Output/" + paraDict['UserName'] + "/" + paraDict['FileOut'] + "/transformer.pth"
        torch.save(model.state_dict(), model_save_path)
        Logging.info(f"正常：模型已保存至 {model_save_path}")

        return all_losses, all_accuracies

    except ImportError as e:
        Logging.info(f"错误：缺少必要的库 - {str(e)}")
        Logging.info("请安装 requirements.txt 中的依赖")
        Logging.info("================程序结束运行================")
        Logging.info(f"总运行时间: {time.time() - TimeStart:.4f} 秒")
        sys.exit(1)
    except Exception as e:
        Logging.info(f"错误：模型训练过程中发生错误 - {str(e)}")
        Logging.info("================程序结束运行================")
        Logging.info(f"总运行时间: {time.time() - TimeStart:.4f} 秒")
        sys.exit(1)

# 测试模型
def transformer_eval(paraDict, all_sequences, all_labels, Logging, TimeStart):
    try:
        Logging.info("正常：准备运行'transformer'模型测试")

        # 创建数据集和数据加载器
        dataset = TensorDataset(all_sequences, all_labels)
        test_loader = DataLoader(dataset, batch_size=int(paraDict['BatchSize']), shuffle=False)

        # 模型实例化
        input_channels = all_sequences.size(1)
        sequence_length = all_sequences.size(2)
        model = TransformerACP(input_channels, sequence_length)

        # 加载模型
        model.load_state_dict(torch.load("backend/Output/" + paraDict['UserName'] + "/" +  paraDict['FileOut'] + "/transformer.pth", weights_only=True))
        model.eval()

        # 定义损失函数
        criterion = nn.BCELoss()

        running_loss = 0.0
        correct = 0
        total = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        
        all_true_labels = []
        all_predicted_scores = []
        TimeTransformer_E = time.time()
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())

                running_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).int()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 计算 TP, FP, TN, FN
                tp += ((predicted == 1) & (labels == 1)).sum().item()
                fp += ((predicted == 1) & (labels == 0)).sum().item()
                tn += ((predicted == 0) & (labels == 0)).sum().item()
                fn += ((predicted == 0) & (labels == 1)).sum().item()

                # 收集真实标签和预测概率
                all_true_labels.extend(labels.cpu().numpy())
                all_predicted_scores.extend(outputs.squeeze().cpu().numpy())

        test_loss = running_loss / len(test_loader)
        test_accuracy = 100 * correct / total
        # 计算 AUC
        auc = roc_auc_score(all_true_labels, all_predicted_scores)
        Logging.info(f"测试丢失率: {test_loss:.4f}, 测试准确率: {test_accuracy:.2f}%")
        Logging.info(f"真阳性 (TP): {tp}")
        Logging.info(f"假阳性 (FP): {fp}")
        Logging.info(f"真阴性 (TN): {tn}")
        Logging.info(f"假阴性 (FN): {fn}")
        Logging.info(f"AUC: {auc:.4f}")
        Logging.info(f"测试耗时: {time.time() - TimeTransformer_E:.4f} 秒")

        return tp, fp, tn, fn, auc

    except ImportError as e:
        Logging.info(f"错误：缺少必要的库 - {str(e)}")
        Logging.info("请安装 requirements.txt 中的依赖")
        Logging.info("================程序结束运行================")
        Logging.info(f"总运行时间: {time.time() - TimeStart:.4f} 秒")
        sys.exit(1)
    except Exception as e:
        Logging.info(f"错误：模型测试过程中发生错误 - {str(e)}")
        Logging.info("================程序结束运行================")
        Logging.info(f"总运行时间: {time.time() - TimeStart:.4f} 秒")
        sys.exit(1)