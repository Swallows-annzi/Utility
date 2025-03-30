import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import os

# 定义CNN模型
class CNNACP(nn.Module):
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
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs, Logging):
    all_losses = []
    all_accuracies = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            # Ensure inputs are tensors
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs)
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
        Logging.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    return all_losses, all_accuracies

# 主函数
def cnn(paraDict, sequences, Logging):
    try:
        Logging.info("正常：准备运行'cnn'模型")
        Pos_sequences, Neg_sequences = sequences

    except ImportError as e:
        Logging.info(f"错误：缺少必要的库 - {str(e)}")
        Logging.info("请安装 requirements.txt 中的依赖")
        Logging.info("================程序结束运行================")
        sys.exit(1)
    except Exception as e:
        Logging.info(f"错误：模型训练过程中发生错误 - {str(e)}")
        Logging.info("================程序结束运行================")
        sys.exit(1)