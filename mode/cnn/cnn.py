import torch
import torch.nn as nn
import torch.nn.functional as F

class ProteinCNN(nn.Module):
    def __init__(self, seq_length, n_amino_acids=20):
        """
            seq_length: 序列最大长度
            n_amino_acids: 氨基酸种类数量 20种
        """
        super(ProteinCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv1d(n_amino_acids, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool1d(2)
        
        # 计算全连接层的输入维度
        self.fc_input_dim = 128 * (seq_length // 8)
        
        # 全连接层
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)  # 二分类输出
        
        # Dropout层防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 卷积块1
        x = self.pool(F.relu(self.conv1(x)))
        
        # 卷积块2
        x = self.pool(F.relu(self.conv2(x)))
        
        # 卷积块3
        x = self.pool(F.relu(self.conv3(x)))
        
        # 展平
        x = x.view(-1, self.fc_input_dim)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)

def train_model(model, train_loader, optimizer, epoch, device, logger):
    """
        list: 该epoch中的所有损失值
        list: 该epoch中的所有准确率
    """
    model.train()
    total = len(train_loader.dataset)
    processed_total = 0
    epoch_losses = []
    epoch_accuracies = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_size = data.size(0)
        processed_total += batch_size
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # 计算当前batch的准确率
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * correct / batch_size
        
        # 记录损失和准确率
        epoch_losses.append(loss.item())
        epoch_accuracies.append(accuracy)
        
        # 计算实际进度
        progress = 100. * processed_total / total
        
        logger.info(
            f'正常：已处理：{processed_total}个样本/共{total}个样本 '
            f'({progress:.0f}%) 损失: {loss.item():.6f}, 准确率: {accuracy:.2f}%'
        )
        
        # 检查损失是否为0
        if loss.item() == 0:
            logger.info("正常：损失已降至0，提前停止训练")
            return epoch_losses, epoch_accuracies
    
    logger.info(f'正常：训练轮次 {epoch} 完成 - 平均损失: {sum(epoch_losses)/len(epoch_losses):.6f}, '
                f'平均准确率: {sum(epoch_accuracies)/len(epoch_accuracies):.2f}%')
    return epoch_losses, epoch_accuracies

def evaluate_model(model, test_loader, device, logger):
    """
        评估模型性能
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    logger.info(
        f'正常：测试集: 平均损失: {test_loss:.4f}, '
        f'正常：准确率 {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)'
    )
    return accuracy
