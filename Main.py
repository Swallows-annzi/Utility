import sys
import configparser 
import logging
import os
from datetime import datetime
import time

#帮助文档
helpStr = '''
帮助文档：
    输入格式：参数名+参数...以此类推
    如果不设置则默认使用配置文件内容

参数名：
--FileIn (String) 设置需要训练的结构文件名
    如果没有设置则抛出错误！

--FileOut (String) 设置输出模型的文件名
    如果没有设置则使用预设文件名：模型名+时间戳下

--MaxLength (Int) 蛋白质序列最大长度
    如果没有设置则使用默认长度
    设置量必须大于等于最大的序列长度
    如果小于最大序列，则自动补全至最长序列长度

--Mode (String) 设置使用模型
    可选用模型: cnn、rnn、mlp、lstm、gcn、gnn、transformer
    若选用多模型比较，则在模型名之间添加'+'例如'cnn+rnn+mlp'
    若不设置则默认选择'cnn'模型

--Epochs (int)
    该项设置训练轮数
    若不设置则默认为 10

--EncodeMode (String) 设置编码方式
    可选用编码方式: One-Hot
    若不设置则默认选择'One-Hot'编码

--BatchSize (int) 设置训练分组数量
    此项决定训练数量分组的大小，并且可能会加速训练
    若过大可能会导致崩溃
    若不设置则默认为 32

--LearningRate (float) 设置学习率
    训练损失下降太慢 - 可以适量增大 LR。
    损失不稳定、震荡 - 需要适量减小 LR。
    若不设置则默认为 0.001

--Function (String) 设置功能
    可选用功能: Train、Eval
    Train: 训练模型
    Eval: 评估模型 - 会将输入的文件作为测试集
    若不设置则默认选择'Train'功能

--UserName (String) 设置用户名
    此项设置用户名，用于区分不同用户提交的模型
    若不设置则默认为 'Swallows_'
    若设置了用户名，则会在 Output 文件夹下创建一个以用户名命名的文件夹分类

--ModeFile (String) 设置模型文件路径
    若使用模型评估'Eval'功能则必须指定此项
    此项设置模型文件路径，用于指定模型文件
    若不指定则会抛出错误
'''
#RNN MLP LSTM CNN Transformer(可能占用大) GCN

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
    TimeReadSequences = time.time()
    # 负样本
    Neg_sequences = []
    # 正样本
    Pos_sequences = []
    try:
        with open(InputFile, 'r') as InF:
            lines = InF.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith(">"):
                    if "NON" in line:
                        Neg_sequences.append(lines[i + 1].strip())
                    else:
                        Pos_sequences.append(lines[i + 1].strip())
                i += 1

        Logging.info(f"正常：成功读取序列文件 {InputFile}")
        Logging.info(f"正常：共读取到 {len(Pos_sequences)} 个正样本序列")
        Logging.info(f"正常：共读取到 {len(Neg_sequences)} 个负样本序列")
        # if len(Pos_sequences) != 0:
        #     Logging.info("正样本序列：")
        #     for i, seq in enumerate(Pos_sequences, 1):
        #         Logging.info(f"序列 {i}: {seq}")
        # if len(Neg_sequences) != 0:
        #     Logging.info("负样本序列：")
        #     for i, seq in enumerate(Neg_sequences, 1):
        #         Logging.info(f"序列 {i}: {seq}")
        Logging.info(f"文件读取耗时: {time.time() - TimeReadSequences:.4f} 秒")
        return Pos_sequences, Neg_sequences

    except FileNotFoundError:
        Logging.info(f"错误：未找到文件 {InputFile}")
        Logging.info("================程序结束运行================")
        Logging.info(f"总运行时间: {time.time() - TimeStart:.4f} 秒")
        sys.exit(1)
    except Exception as e:
        Logging.info(f"错误：读取文件时发生错误 - {str(e)}")
        Logging.info("================程序结束运行================")
        Logging.info(f"总运行时间: {time.time() - TimeStart:.4f} 秒")
        sys.exit(1)

#统一序列长度
def StandardizeSequences(sequences, paraDict):
    TimeStandardizeSequences = time.time()
    MaxLength = int(paraDict['MaxLength'])
    VoidDict = paraDict['VoidDict']
    Pos_sequences, Neg_sequences = sequences
    # 获取序列的最大长度
    SequencesMaxLength = max(len(line) for line in (Pos_sequences + Neg_sequences))
    if SequencesMaxLength > MaxLength:
        Logging.info(f"警告：当前序列最大长度为 {SequencesMaxLength} 大于设置数量 {MaxLength}")
        paraDict['MaxLength'] = SequencesMaxLength
        Logging.info(f"正常：已修改最大速度为 {SequencesMaxLength}")
    else:
        Logging.info(f"正常：当前序列最大长度为 {SequencesMaxLength}")
    
    # 使用目标长度
    TargetLength = max(SequencesMaxLength, MaxLength)
    
    # 标准化序列长度
    Pos_sequences = [line.ljust(TargetLength, VoidDict) for line in Pos_sequences]
    Neg_sequences = [line.ljust(TargetLength, VoidDict) for line in Neg_sequences]
    
    Logging.info(f"正常：所有序列已标准化为长度 {TargetLength}")
    Logging.info(f"正常：共读修改 {len(Pos_sequences)} 个正样本序列")
    Logging.info(f"正常：共读修改 {len(Neg_sequences)} 个负样本序列")
    # if len(Pos_sequences) != 0:
    #     Logging.info("已修改正样本序列：")
    #     for i, seq in enumerate(Pos_sequences, 1):
    #         Logging.info(f"序列 {i}: {seq}")
    # if len(Neg_sequences) != 0:
    #     Logging.info("已修改负样本序列：")
    #     for i, seq in enumerate(Neg_sequences, 1):
    #         Logging.info(f"序列 {i}: {seq}")
    Logging.info(f"统一长度耗时: {time.time() - TimeStandardizeSequences:.4f} 秒")
    return Pos_sequences, Neg_sequences

#One-hot编码
def One_Hot(sequences, paraDict, Logging):
    import torch
    TimeOne_Hot = time.time()
    aa_dict = paraDict['AADict']
    Logging.info("正常：开始序列编码")
    Length = int(paraDict['MaxLength'])
    
    encoded_sequences = []
    for seq in sequences:
        encoded = torch.zeros(len(aa_dict), Length)
        for i, aa in enumerate(seq):
            if aa in aa_dict:
                idx = aa_dict.index(aa)
                encoded[idx][i] = 1
        encoded_sequences.append(encoded)
    
    Logging.info(f"正常：序列编码完成，形状为 [{len(sequences)}, {len(aa_dict)}, {Length}], 耗时 {time.time() - TimeOne_Hot:.4f} 秒")
    return torch.stack(encoded_sequences)

#线型图生成
def plot_training_progress(paraDict, graphs):
    """
        graphs: 包含每个模型的训练损失和准确率数据的字典
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        modes = paraDict['Mode'].split('+')
        FlieOut = "Output/" + paraDict['UserName'] + "/" + paraDict['FileOut'] + "/" + '-'.join(modes) + ".png"

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(12, 5))

        max_epochs = 0
        # 损失值子图
        plt.subplot(1, 2, 1)
        colors_loss = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for i, mode in enumerate(modes):
            if mode in graphs and graphs[mode]:
                losses, _ = graphs[mode]
                epochs = np.arange(1, len(losses) + 1)
                max_epochs = max(max_epochs, len(losses))
                plt.plot(epochs, losses, f'{colors_loss[i % len(colors_loss)]}-', label=f'{mode}', marker='o', markersize=4, 
                        markerfacecolor='white', markeredgecolor=colors_loss[i % len(colors_loss)], linewidth=2)
        epochs = np.arange(1, max_epochs + 1)
        plt.title('训练损失变化曲线')
        plt.xlabel('训练次数')
        plt.xticks(epochs)
        plt.ylabel('损失值')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # 准确率子图
        plt.subplot(1, 2, 2)
        colors_acc = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        max_epochs = 0
        for i, mode in enumerate(modes):
            if mode in graphs and graphs[mode]:
                _, accuracies = graphs[mode]
                epochs = np.arange(1, len(accuracies) + 1)
                max_epochs = max(max_epochs, len(accuracies))
                plt.plot(epochs, accuracies, f'{colors_acc[i % len(colors_acc)]}-', label=f'{mode}', marker='o', markersize=4,
                        markerfacecolor='white', markeredgecolor=colors_acc[i % len(colors_acc)], linewidth=2)
        epochs = np.arange(1, max_epochs + 1)
        plt.title('训练准确率变化曲线')
        plt.xlabel('训练次数')
        plt.xticks(epochs)
        plt.ylabel('准确率 (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.tight_layout()
        plt.savefig(FlieOut, dpi=300, bbox_inches='tight')
        plt.close()

        Logging.info(f"正常：训练进度图已保存至 {FlieOut}")

    except ImportError:
        Logging.info("警告：缺少matplotlib库，无法生成训练进度图表")
    except Exception as e:
        Logging.info(f"警告：生成训练进度图表时发生错误 - {str(e)}")

def main():

    if '-h' in sys.argv or '--help' in sys.argv:
        Logging.info(helpStr)
        Logging.info("================程序运行结束================")
        Logging.info(f"总运行时间: {time.time() - TimeStart:.4f} 秒")
        sys.exit(0)

    #参数初始化
    paraDict = {
        #模型名
        'UserName': "Swallows_",
        #序列化设置
        'MaxLength':10,
        'AADict':"ACDEFGHIKLMNPQRSTVWY",
        'VoidDict':"X",
        #学习率
        'LearningRate': 0.001,
        #模型选用
        'Mode':"cnn",
        #模型功能
        'Function':"Train",
        #训练分组数量
        'BatchSize': 32,
        #编码方式
        'EncodeMode':"one-hot",
        #训练轮数
        'Epochs': 10,
        #输入输出文件
        'FileIn': None,
        'FileOut': f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
        'ModeFile': "Output/"
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
                Logging.info(f"总运行时间: {time.time() - TimeStart:.4f} 秒")
                sys.exit(1)
        else:
            if currPara is None:
                continue
            paraDict[currPara] = para
            Logging.info(f"正常：参数 {currPara} 设置为：{para}")
            currPara = None
    if len(sys.argv) == 1:
        Logging.info("警告：无设置参数")
    Logging.info("正常：已设置参数列表")
    for key, value in paraDict.items():
        Logging.info(f"{key} = {value}")
    Logging.info("正常：参数设置完成")

    #参数检查
    if paraDict['FileIn'] is None:
        Logging.info("错误：未设置输入文件")
        Logging.info("使用 -h 或 --help 查看帮助文档")
        Logging.info("================程序运行结束================")
        Logging.info(f"总运行时间: {time.time() - TimeStart:.4f} 秒")
        sys.exit(1)

    modes = paraDict['Mode'].split('+')

    if any(str not in ['cnn', 'rnn', 'mlp', 'lstm', 'gcn', 'gnn', 'transformer'] for str in modes):
        Logging.info("错误：未知模型")
        Logging.info("使用 -h 或 --help 查看帮助文档")
        Logging.info("================程序运行结束================")
        Logging.info(f"总运行时间: {time.time() - TimeStart:.4f} 秒")
        sys.exit(1)

    #读取序列并标准化序列长度
    sequences = StandardizeSequences(ReadSequences(paraDict['FileIn']), paraDict, )

    if paraDict['Function'] not in "Eval" and paraDict['Function'] not in "Train":
        Logging.info("错误：未知功能")
        Logging.info("使用 -h 或 --help 查看帮助文档")
        Logging.info("================程序运行结束================")
        Logging.info(f"总运行时间: {time.time() - TimeStart:.4f} 秒")
        sys.exit(1)
    
    if not os.path.exists('Output/' + paraDict['UserName'] + '/' + paraDict['FileOut']):
        os.makedirs('Output/' + paraDict['UserName'] + '/' + paraDict['FileOut'])
    
    # 提取的代码插入位置
    Pos_sequences, Neg_sequences = sequences
    import torch
    # if len(Pos_sequences) != 0 and len(Neg_sequences) != 0:
    Pos_sequences = One_Hot(Pos_sequences, paraDict, Logging)
    Neg_sequences = One_Hot(Neg_sequences, paraDict, Logging)
    if isinstance(Pos_sequences, list):
        Pos_sequences = torch.stack(Pos_sequences) if Pos_sequences else torch.tensor([])
    if isinstance(Neg_sequences, list):
        Neg_sequences = torch.stack(Neg_sequences) if Neg_sequences else torch.tensor([])

    all_sequences = torch.cat((Pos_sequences, Neg_sequences), dim=0)
    all_labels = torch.cat((torch.ones(Pos_sequences.size(0)), torch.zeros(Neg_sequences.size(0))), dim=0)
    Logging.info("正常：已合并正负样本")

    # elif len(Pos_sequences) == 0 and len(Neg_sequences) != 0:
    #     Neg_sequences = One_Hot(Neg_sequences, paraDict, Logging)
    #     if isinstance(Neg_sequences, list):
    #         Neg_sequences = torch.stack(Neg_sequences) if Neg_sequences else torch.tensor([])
    #
    #     all_sequences = torch.cat(Neg_sequences, dim=0)
    #     all_labels = torch.cat(torch.zeros(Neg_sequences.size(0)), dim=0)
    #     Logging.info("正常：已合并正负样本")
    #
    # elif len(Pos_sequences) != 0 and len(Neg_sequences) == 0:
    #     Pos_sequences = One_Hot(Pos_sequences, paraDict, Logging)
    #     if isinstance(Pos_sequences, list):
    #         Pos_sequences = torch.stack(Pos_sequences) if Pos_sequences else torch.tensor([])
    #
    #     all_sequences = torch.cat(Pos_sequences, dim=0)
    #     all_labels = torch.cat(torch.ones(Pos_sequences.size(0)), dim=0)
    #     Logging.info("正常：已合并正负样本")

    
    modes = paraDict['Mode'].split('+')

    graphs = {
        'cnn': "",
        'rnn': "",
        'mlp': "",
        'lstm': "",
        'gcn': "",
        'gnn': "",
        'transformer': ""
    }
    
    if "cnn" in modes:
        if paraDict['Function'] in "Train":
            from mode.cnn.cnn import cnn_train
            graphs['cnn'] = cnn_train(paraDict, all_sequences, all_labels, Logging, TimeStart)
        if paraDict['Function'] in "Eval":
            from mode.cnn.cnn import cnn_eval
            cnn_eval(paraDict, all_sequences, all_labels, Logging, TimeStart)
    if "rnn" in modes:
        if paraDict['Function'] in "Train":
            from mode.rnn.rnn import rnn_train
            graphs['rnn'] = rnn_train(paraDict, all_sequences, all_labels, Logging, TimeStart)
        if paraDict['Function'] in "Eval":
            from mode.rnn.rnn import rnn_eval
            rnn_eval(paraDict, all_sequences, all_labels, Logging, TimeStart)
    if "mlp" in modes:
        if paraDict['Function'] in "Train":
            from mode.mlp.mlp import mlp_train
            graphs['mlp'] = mlp_train(paraDict, all_sequences, all_labels, Logging, TimeStart)
        if paraDict['Function'] in "Eval":
            from mode.mlp.mlp import mlp_eval
            mlp_eval(paraDict, all_sequences, all_labels, Logging, TimeStart)
    if "lstm" in modes:
        if paraDict['Function'] in "Train":
            from mode.lstm.lstm import lstm_train
            graphs['lstm'] = lstm_train(paraDict, all_sequences, all_labels, Logging, TimeStart)
        if paraDict['Function'] in "Eval":
            from mode.lstm.lstm import lstm_eval
            lstm_eval(paraDict, all_sequences, all_labels, Logging, TimeStart)
    if "transformer" in modes:
        if paraDict['Function'] in "Train":
            from mode.transformer.transformer import transformer_train
            graphs['transformer'] = transformer_train(paraDict, all_sequences, all_labels, Logging, TimeStart)
        if paraDict['Function'] in "Eval":
            from mode.transformer.transformer import transformer_eval
            transformer_eval(paraDict, all_sequences, all_labels, Logging, TimeStart)
    # if "gcn" in modes:
    #     if paraDict['Function'] in "Train":
    #         from mode.gcn.gcn import gcn_train
    #     if paraDict['Function'] in "Eval":
    #         from mode.gcn.gcn import gcn_eval
    # if "gnn" in modes:
    #     if paraDict['Function'] in "Train":
    #         from mode.gnn.gnn import gnn_train
    #     if paraDict['Function'] in "Eval":
    #         from mode.gnn.gnn import gnn_eval

    plot_training_progress(paraDict, graphs)

if __name__ == '__main__':
    TimeStart = time.time()
    Logging = logger()
    Logging.info("================程序开始运行================")
    # Config = config()
    main()
    Logging.info("================程序结束运行================")
    Logging.info(f"总运行时间: {time.time() - TimeStart:.4f} 秒")