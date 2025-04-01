import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from Main import One_Hot, plot_training_progress
import time

# 定义CNN模型
class CNNACP(nn.Module):
    """
    定义一维卷积神经网络模型。

    参数:
    input_channels (int): 输入数据的通道数。
    sequence_length (int): 输入序列的长度。
    """
    def __init__(self, input_channels, sequence_length):
        super(CNNACP, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(32 * (sequence_length // 4), 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播函数。

        参数:
        x (torch.Tensor): 输入数据。

        返回:
        torch.Tensor: 模型输出。
        """
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs, Logging):
    """
    训练模型。

    参数:
    model (nn.Module): 待训练的模型。
    train_loader (DataLoader): 训练数据加载器。
    criterion (nn.Module): 损失函数。
    optimizer (optim.Optimizer): 优化器。
    num_epochs (int): 训练轮数。
    Logging: 日志记录器。

    返回:
    all_losses: 每一轮的损失值列表。
    all_accuracies: 每一轮的准确率列表。
    """
    all_losses = []
    all_accuracies = []
    for epoch in range(num_epochs):
        TimeCNNNum = time.time()
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
        Logging.info(f'训练次数: {epoch + 1}/{num_epochs}, 丢失率: {epoch_loss:.4f}, 准确率: {epoch_accuracy:.2f}%, 耗时: {time.time() - TimeCNNNum:.4f} 秒')

    return all_losses, all_accuracies

# 主函数
def cnn_train(paraDict, sequences, Logging, TimeStart):
    """
    主函数，用于运行CNN模型训练。

    参数:
    paraDict (dict): 包含模型参数的字典。
    sequences (tuple): 包含正样本和负样本序列的元组。
    Logging: 日志记录器。
    """
    try:
        Logging.info("正常：准备运行'cnn'模型")
        Pos_sequences, Neg_sequences = sequences

        if len(Pos_sequences) != 0:
            Pos_sequences = One_Hot(Pos_sequences, paraDict, Logging)
        if len(Neg_sequences) != 0:
            Neg_sequences = One_Hot(Neg_sequences, paraDict, Logging)
        # 合并正负样本
        all_sequences = torch.cat((Pos_sequences, Neg_sequences), dim=0)
        all_labels = torch.cat((torch.ones(Pos_sequences.size(0)), torch.zeros(Neg_sequences.size(0))), dim=0)
        Logging.info("正常：已合并正负样本")

        # 创建数据集和数据加载器
        dataset = TensorDataset(all_sequences, all_labels)
        train_loader = DataLoader(dataset, batch_size=int(paraDict['BatchSize']), shuffle=True)

        # 模型实例化
        input_channels = all_sequences.size(1)
        sequence_length = all_sequences.size(2)
        model = CNNACP(input_channels, sequence_length)

        # 定义损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=paraDict['LearningRate'])

        # 训练模型
        num_epochs = int(paraDict['Epochs'])  # 修改此处
        TimeCNN_T = time.time()
        all_losses, all_accuracies = train_model(model, train_loader, criterion, optimizer, num_epochs, Logging)
        Logging.info(f"总训练耗时: {time.time() - TimeCNN_T:.4f} 秒")

        # 绘制训练进度图
        plot_training_progress(all_losses, all_accuracies, paraDict, Logging)

        # 保存模型
        model_save_path = "Output/" + paraDict['UserName'] + "/" + paraDict['Mode'] + paraDict['FileOut'] + ".pth"
        torch.save(model.state_dict(), model_save_path)
        Logging.info(f"正常：模型已保存至 {model_save_path}")

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
def cnn_eval(paraDict, test_sequences, Logging, TimeStart):
    """
    测试模型。

    参数:
    paraDict (dict): 包含模型参数的字典。
    test_sequences (tuple): 包含正样本和负样本序列的元组。
    Logging: 日志记录器。
    """
    try:
        Logging.info("正常：准备运行模型测试")
        Pos_sequences, Neg_sequences = test_sequences

        if len(Pos_sequences) != 0:
            Pos_sequences = One_Hot(Pos_sequences, paraDict, Logging)
        if len(Neg_sequences) != 0:
            Neg_sequences = One_Hot(Neg_sequences, paraDict, Logging)
        # 合并正负样本
        all_sequences = torch.cat((Pos_sequences, Neg_sequences), dim=0)
        all_labels = torch.cat((torch.ones(Pos_sequences.size(0)), torch.zeros(Neg_sequences.size(0))), dim=0)
        Logging.info("正常：已合并正负样本")

        # 创建数据集和数据加载器
        dataset = TensorDataset(all_sequences, all_labels)
        test_loader = DataLoader(dataset, batch_size=int(paraDict['BatchSize']), shuffle=False)  # 修改此处

        # 模型实例化
        input_channels = all_sequences.size(1)
        sequence_length = all_sequences.size(2)
        model = CNNACP(input_channels, sequence_length)

        # 加载模型
        model.load_state_dict(torch.load(paraDict['ModeFile'], weights_only=True))
        model.eval()

        # 定义损失函数
        criterion = nn.BCELoss()

        running_loss = 0.0
        correct = 0
        total = 0
        TimeCNN_E = time.time()
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())

                running_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).int()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss = running_loss / len(test_loader)
        test_accuracy = 100 * correct / total
        Logging.info(f"测试丢失率: {test_loss:.4f}, 测试准确率: {test_accuracy:.2f}%")
        Logging.info(f"测试耗时: {time.time() - TimeCNN_E:.4f} 秒")

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