import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

# 定义 Transformer 模型
class TransformerACP(nn.Module):
    """
    定义 Transformer 神经网络模型。

    参数:
    input_channels (int): 输入数据的通道数。
    sequence_length (int): 输入序列的长度。
    nhead (int): 多头注意力机制的头数。
    num_layers (int): Transformer 编码器层数。
    """
    def __init__(self, input_channels, sequence_length, nhead=4, num_layers=2):
        super(TransformerACP, self).__init__()
        self.embedding = nn.Linear(input_channels, 64)
        self.positional_encoding = self._generate_positional_encoding(sequence_length, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(64 * sequence_length, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
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
    """
    主函数，用于运行 Transformer 模型训练。

    参数:
    paraDict (dict): 包含模型参数的字典。
    all_sequences (torch.Tensor): 合并后的序列数据。
    all_labels (torch.Tensor): 合并后的标签数据。
    Logging: 日志记录器。
    TimeStart: 程序开始时间。
    """
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
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=float(paraDict['LearningRate']))

        # 训练模型
        num_epochs = int(paraDict['Epochs'])
        TimeTransformer_T = time.time()
        all_losses, all_accuracies = train_model(model, train_loader, criterion, optimizer, num_epochs, Logging)
        Logging.info(f"总训练耗时: {time.time() - TimeTransformer_T:.4f} 秒")

        # 保存模型
        model_save_path = "Output/" + paraDict['UserName'] + "/" + paraDict['FileOut'] + "/transformer-" + paraDict['FileOut'] + ".pth"
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
    """
    测试模型。

    参数:
    paraDict (dict): 包含模型参数的字典。
    all_sequences (torch.Tensor): 合并后的序列数据。
    all_labels (torch.Tensor): 合并后的标签数据。
    Logging: 日志记录器。
    TimeStart: 程序开始时间。
    """
    try:
        Logging.info("正常：准备运行模型测试")

        # 创建数据集和数据加载器
        dataset = TensorDataset(all_sequences, all_labels)
        test_loader = DataLoader(dataset, batch_size=int(paraDict['BatchSize']), shuffle=False)

        # 模型实例化
        input_channels = all_sequences.size(1)
        sequence_length = all_sequences.size(2)
        model = TransformerACP(input_channels, sequence_length)

        # 加载模型
        model.load_state_dict(torch.load(paraDict['ModeFile'], weights_only=True))
        model.eval()

        # 定义损失函数
        criterion = nn.BCELoss()

        running_loss = 0.0
        correct = 0
        total = 0
        TimeTransformer_E = time.time()
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
        Logging.info(f"测试耗时: {time.time() - TimeTransformer_E:.4f} 秒")

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