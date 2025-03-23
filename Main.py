import sys
import configparser 
import logging
import os
from datetime import datetime

#帮助文档
helpStr = '''
帮助文档：
    输入格式：参数名+参数...以此类推
    如果不设置则默认使用配置文件内容

参数名：
--fileIn (String) 设置需要训练的结构文件名
    如果没有设置则抛出错误！

--fileOut (String) 设置输出模型的文件名
    如果没有设置则使用预设文件名：模型名+时间戳

--MaxLength (Int) 蛋白质序列最大长度
    如果没有设置则使用默认长度
    设置量必须大于等于最大的序列长度
    如果小于最大序列，则自动补全至最长序列长度

--Mode (String) 设置使用模型
    可选用模型: cnn
    若不设置则默认选择'cnn'模型

--num_epochs (int)
    该项设置训练轮数
    若不设置则默认为 10
'''

#日志记录
#单次运行的日志记录
def logger():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    LogFileName = f"logs/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LogFileName, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging

#配置文件
def config():
    config = configparser.ConfigParser()
    if config.read('Config.cfg'):
        Logging.info("正常：检测到配置文件 Config.cfg")
    else:
        # 如果文件不存在，创建默认配置
        config['File'] = {
            'x': 'null'
        }
        with open('Config.cfg', 'w', encoding='utf-8') as ConfigFile:
            config.write(ConfigFile)
        Logging.info("警告：未检测到配置文件 Config.cfg")
        Logging.info("正常：已生成默认的配置文件 Config.cfg")
    return {
        'x': config.get('File', 'x')
    }

#读取序列
def ReadSequences(InputFile):
    sequences = []
    try:
        with open(InputFile, 'r') as f:
            CurrentSeq = ''
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if CurrentSeq:
                        sequences.append(CurrentSeq)
                        CurrentSeq = ''
                else:
                    CurrentSeq += line
            if CurrentSeq:
                sequences.append(CurrentSeq)
        
        Logging.info(f"正常：成功读取序列文件 {InputFile}")
        Logging.info("输入序列：")
        for i, seq in enumerate(sequences, 1):
            Logging.info(f"序列 {i}: {seq}")
        Logging.info(f"正常：共读取到 {len(sequences)} 个序列")
        return sequences
    
    except FileNotFoundError:
        Logging.info(f"错误：未找到文件 {InputFile}")
        sys.exit(1)
    except Exception as e:
        Logging.info(f"错误：读取文件时发生错误 - {str(e)}")
        sys.exit(1)

#统一序列长度
def StandardizeSequences(sequences, MaxLength, VoidDict):
    # 获取序列的最大长度
    current_max_len = max(len(seq) for seq in sequences)
    if current_max_len > MaxLength:
        Logging.info(f"警告：当前序列最大长度为 {current_max_len}")
    else:
        Logging.info(f"正常：当前序列最大长度为 {current_max_len}")
    
    # 使用目标长度
    target_length = max(current_max_len, MaxLength)
    
    # 标准化序列长度
    newsequences = []
    for seq in sequences:
        if len(seq) < target_length:
            newsequences.append(seq + VoidDict * (target_length - len(seq)))
        else:
            newsequences.append(seq)
    
    Logging.info(f"正常：所有序列已标准化为长度 {target_length}")
    Logging.info("已修改序列：")
    for i, seq in enumerate(newsequences, 1):
        Logging.info(f"序列 {i}: {seq}")
    return newsequences

#One-hot编码
def EncodeSequences(sequences, aa_dict):
    import torch
    Logging.info("正常：开始序列编码")

    target_length = max(len(seq) for seq in sequences)
    
    encoded_sequences = []
    for seq in sequences:
        # 检查序列长度
        if len(seq) != target_length:
            Logging.info(f"错误：序列长度不一致 - 预期 {target_length}，实际 {len(seq)}")
            Logging.info("================程序结束运行================")
            sys.exit(1)
            
        encoded = torch.zeros(len(aa_dict), target_length)
        for i, aa in enumerate(seq):
            if aa in aa_dict:
                idx = aa_dict.index(aa)
                encoded[idx][i] = 1
            #对于VoidDict字符，保持为0向量
        
        encoded_sequences.append(encoded)
    
    Logging.info(f"正常：序列编码完成，形状为 [{len(sequences)}, {len(aa_dict)}, {target_length}]")
    return torch.stack(encoded_sequences)

#线型图生成
def plot_training_progress(losses, accuracies, save_path, mode):
    """
        losses: 每个epoch的损失值列表
        accuracies: 每个epoch的准确率列表
        save_path: 图表保存路径
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np  # 新增导入，用于设置刻度
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(12, 5))
        
        # 生成轮次列表
        epochs = np.arange(1, len(losses) + 1)
        
        # 损失值子图
        plt.subplot(1, 2, 1)
        plt.plot(epochs, losses, 'b-', label='训练损失', marker='o', markersize=4, 
                markerfacecolor='white', markeredgecolor='blue', linewidth=2)
        plt.title(mode + '训练损失变化曲线')
        plt.xlabel('训练轮次')
        plt.xticks(epochs)  # 设置 x 轴刻度为整数
        plt.ylabel('损失值')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 准确率子图
        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracies, 'r-', label='训练准确率', marker='o', markersize=4,
                markerfacecolor='white', markeredgecolor='red', linewidth=2)
        plt.title(mode + '训练准确率变化曲线')
        plt.xlabel('训练轮次')
        plt.xticks(epochs)  # 设置 x 轴刻度为整数
        plt.ylabel('准确率 (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        Logging.info(f"正常：训练进度图已保存至 {save_path}")
        
    except ImportError:
        Logging.info("警告：缺少matplotlib库，无法生成训练进度图表")
    except Exception as e:
        Logging.info(f"警告：生成训练进度图表时发生错误 - {str(e)}")

def cnn(paraDict, sequences, Logging):
    """
    :param paraDict: 包含参数配置的字典
    :param sequences: 标准化后的序列数据
    :param Logging: 日志记录器对象
    """
    Logging.info("正常：准备运行'cnn'模型")
    try:
        import torch
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        from mode.cnn.cnn import ProteinCNN, train_model, evaluate_model

        # 设置设备
        num_epochs = paraDict['num_epochs']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Logging.info(f"正常：使用设备 {device}")

        # 序列编码
        X = EncodeSequences(sequences, paraDict['AADict'])
        y = torch.ones(len(sequences), dtype=torch.long)

        # 创建数据集
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 初始化模型
        model = ProteinCNN(
            seq_length=len(sequences[0]),
            n_amino_acids=len(paraDict['AADict'])
        ).to(device)

        # 设置优化器
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        Logging.info(f"正常：开始模型训练，训练轮次: {num_epochs}")

        # 记录训练数据
        all_losses = []
        all_accuracies = []

        for epoch in range(1, num_epochs + 1):
            Logging.info(f"正常：开始第 {epoch} 轮训练")
            epoch_losses, epoch_accuracies = train_model(model, train_loader, optimizer, epoch, device, Logging)

            all_losses.extend(epoch_losses)
            all_accuracies.extend(epoch_accuracies)

            if epoch_losses[-1] == 0:  # 如果最后一个 batch 的损失为 0，提前停止训练
                break

        # 生成训练进度图表
        plot_path = f"Output/{paraDict['Mode']}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
        plot_training_progress(all_losses, all_accuracies, plot_path, paraDict['Mode'])

        # 保存模型
        torch.save(model.state_dict(), "Output/" + paraDict['Mode'] + "_" + paraDict['fileOut'])
        Logging.info(f"正常：模型已保存至 Output/{paraDict['Mode']}_{paraDict['fileOut']}")

    except ImportError as e:
        Logging.info(f"错误：缺少必要的库 - {str(e)}")
        Logging.info("请安装 requirements.txt 中的依赖")
        Logging.info("================程序结束运行================")
        sys.exit(1)
    except Exception as e:
        Logging.info(f"错误：模型训练过程中发生错误 - {str(e)}")
        Logging.info("================程序结束运行================")
        sys.exit(1)

def main():

    if '-h' in sys.argv or '--help' in sys.argv:
        Logging.info(helpStr)
        Logging.info("================程序运行结束================")
        sys.exit(0)

    #参数初始化
    paraDict = {
        #序列化设置
        'MaxLength':10,
        'AADict':"ACDEFGHIKLMNPQRSTVWY",
        'VoidDict':"X",
        #模型选用
        'Mode':"cnn",
        #训练轮数
        'num_epochs': 10,
        #输入输出文件定义
        'fileIn': None,
        'fileOut': f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pth",
    }

    #参数写入
    Logging.info("正常：开始配置参数")
    currPara = None
    for para in sys.argv:
        if para.startswith('--'):
            currPara = para[2:]
            if currPara not in paraDict:
                Logging.info(f"错误：未知参数 '{para}'")
                Logging.info("使用 -h 或 --help 查看帮助文档")
                Logging.info("================程序运行结束================")
                sys.exit(1)
        else:
            if currPara is None:
                continue
            paraDict[currPara] = para
            Logging.info(f"正常：参数 {currPara} 设置为：{para}")
            currPara = None
    if len(sys.argv):
        Logging.info("警告：无设置参数")
    Logging.info("正常：已设置参数列表")
    for key, value in paraDict.items():
        Logging.info(f"{key} = {value}")
    Logging.info("正常：参数设置完成")

    #参数检查
    if paraDict['fileIn'] is None:
        Logging.info("错误：未设置输入文件")
        Logging.info("使用 -h 或 --help 查看帮助文档")
        Logging.info("================程序运行结束================")
        sys.exit(1)

    #读取序列并标准化序列长度
    sequences = StandardizeSequences(ReadSequences(paraDict['fileIn']), int(paraDict['MaxLength']), paraDict['VoidDict'])

    if not os.path.exists('Output'):
        os.makedirs('Output')

    if paraDict['Mode'] in "cnn":
        cnn(paraDict, sequences, Logging)

if __name__ == '__main__':
    Logging = logger()
    Logging.info("================程序开始运行================")
    # Config = config()
    main()
    Logging.info("================程序结束运行================")