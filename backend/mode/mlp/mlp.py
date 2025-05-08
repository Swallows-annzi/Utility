import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from sklearn.metrics import roc_auc_score

# 定义 MLP 模型
class MLPACP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPACP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播函数。

        参数:
        x (torch.Tensor): 输入数据。

        返回:
        torch.Tensor: 模型输出。
        """
        x = x.view(x.size(0), -1)  # 展平输入数据
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs, Logging):
    all_losses = []
    all_accuracies = []
    for epoch in range(num_epochs):
        TimeMLPNum = time.time()
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
            predicted = (outputs.squeeze() > 0.5).int()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        all_losses.append(epoch_loss)
        all_accuracies.append(epoch_accuracy)
        Logging.info(f'训练次数: {epoch + 1}/{num_epochs}, 丢失率: {epoch_loss:.4f}, 准确率: {epoch_accuracy:.2f}%, 耗时: {time.time() - TimeMLPNum:.4f} 秒')

    return all_losses, all_accuracies

# 主函数
def mlp_train(paraDict, all_sequences, all_labels, Logging, TimeStart):
    try:
        Logging.info("正常：准备运行'mlp'模型")

        # 创建数据集和数据加载器
        dataset = TensorDataset(all_sequences, all_labels)
        train_loader = DataLoader(dataset, batch_size=int(paraDict['BatchSize']), shuffle=True)

        # 模型实例化
        input_size = all_sequences.view(all_sequences.size(0), -1).size(1)
        hidden_size = 128
        model = MLPACP(input_size, hidden_size)

        # 定义损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=float(paraDict['LearningRate']))

        # 训练模型
        num_epochs = int(paraDict['Epochs'])
        TimeMLP_T = time.time()
        all_losses, all_accuracies = train_model(model, train_loader, criterion, optimizer, num_epochs, Logging)
        Logging.info(f"总训练耗时: {time.time() - TimeMLP_T:.4f} 秒")

        # 保存模型
        model_save_path = "backend/Output/" + paraDict['UserName'] + "/" + paraDict['FileOut'] + "/mlp.pth"
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
def mlp_eval(paraDict, all_sequences, all_labels, Logging, TimeStart):
    """
    测试模型。

    参数:
    paraDict (dict): 包含模型参数的字典。
    test_sequences (tuple): 包含正样本和负样本序列的元组。
    Logging: 日志记录器。
    """
    try:
        Logging.info("正常：准备运行'mlp'模型测试")

        # 创建数据集和数据加载器
        dataset = TensorDataset(all_sequences, all_labels)
        test_loader = DataLoader(dataset, batch_size=int(paraDict['BatchSize']), shuffle=False)

        # 模型实例化
        input_size = all_sequences.view(all_sequences.size(0), -1).size(1)
        hidden_size = 128
        model = MLPACP(input_size, hidden_size)

        # 加载模型
        checkpoint = torch.load("backend/Output/" + paraDict['UserName'] + "/" +  paraDict['FileOut'] + "/mlp.pth", weights_only=True)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
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
        TimeMLP_E = time.time()
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
        Logging.info(f"测试耗时: {time.time() - TimeMLP_E:.4f} 秒")

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