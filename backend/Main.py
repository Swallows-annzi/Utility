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
            如果没有设置则使用预设文件名：时间戳

        --MaxLength (Int) 蛋白质序列最大长度
            如果没有设置则使用默认长度
            设置量必须大于等于最大的序列长度
            如果小于最大序列，则自动补全至最长序列长度

        --Mode (String) 设置使用模型
            可选用模型: cnn、rnn、mlp、lstm、transformer
            若选用多模型比较，则在模型名之间添加'+'例如'cnn+rnn+mlp'
            若不设置则默认选择'cnn'模型

        --Epochs (int) 设置训练轮数
            该项设置训练轮数
            若不设置则默认为 10
        
        --Test (float) 设置测试集比例
            该项设置测试集比例
            若不设置则默认为 0.2
            若设置了测试集比例，则会在 Output 文件夹下创建一个测试集

        --BatchSize (int) 设置训练分组数量
            此项决定训练数量分组的大小，并且可能会加速训练
            若过大可能会导致崩溃
            若不设置则默认为 32

        --LearningRate (float) 设置学习率
            训练损失下降太慢 - 可以适量增大 LR。
            损失不稳定、震荡 - 需要适量减小 LR。
            若不设置则默认为 0.001

        --UserName (String) 设置用户名
            此项设置用户名，用于区分不同用户提交的模型
            若不设置则默认为 'Swallows_'
            若设置了用户名，则会在 Output 文件夹下创建一个以用户名命名的文件夹分类
'''

#单次运行的日志记录
def logger():
    if not os.path.exists('backend/logs'):
        os.makedirs('backend/logs')
    LogFileName = f"backend/logs/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
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
def Config(paraDict):
    config = configparser.ConfigParser()
    config['Parameters'] = {}
    for key, value in paraDict.items():
        config['Parameters'][key] = str(value)
    
    output_dir = os.path.join('backend/Output', paraDict['UserName'], paraDict['FileOut'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    config_path = os.path.join(output_dir, 'Config.cfg')
    with open(config_path, 'w', encoding='utf-8') as config_file:
        config.write(config_file)
    
    Logging.info(f"正常：已将参数保存至配置文件 {config_path}")

# 读取配置文件
def ReadConfig(paraDict):
    config = configparser.ConfigParser()
    output_dir = os.path.join('backend/Output', paraDict['UserName'], paraDict['ModeFile'])
    config_path = os.path.join(output_dir, 'Config.cfg')
    if config.read(config_path):
        Logging.info(f"正常：成功读取配置文件 {config_path}")
    else:
        Logging.info(f"警告：未找到配置文件 {config_path}")
        Logging.info("================程序结束运行================")
        Logging.info(f"总运行时间: {time.time() - TimeStart:.4f} 秒")
        sys.exit(1)
    for key, value in config['Parameters'].items():
        Logging.info(f"{key} = {value}")
    return config['Parameters']

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
        Logging.info(f"正常：已修改最大长度为 {SequencesMaxLength}")
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
    Logging.info(f"类别分布 - 正样本: {len(Pos_sequences)}, 负样本: {len(Neg_sequences)}, 总样本: {len(Pos_sequences) + len(Neg_sequences)}, 正负样本比重: {len(Pos_sequences) / len(Neg_sequences):.2f}(正/负)样本比例")
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
        FlieOut = "backend/Output/" + paraDict['UserName'] + "/" + paraDict['FileOut'] + "/" + '-'.join(modes) + ".png"

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

# 处理与合并正负样本
def sample(paraDict, FileIn):
    # 读取序列并标准化序列长度
    sequences = StandardizeSequences(ReadSequences(FileIn), paraDict)
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

    return all_sequences, all_labels

# 保存评估指标到 JSON 文件
def cmc(paraDict, TrainGraphs, EvalGraphs):
    import json
    output_dir = os.path.join('backend/Output', paraDict['UserName'], paraDict['FileOut'])
    
    metrics_file = os.path.join(output_dir, 'cmc.json')
    metrics_data = {}

    for key , value in paraDict.items():
        metrics_data[key] = value

    for mode, data in TrainGraphs.items():
        if data:
            all_losses, all_accuracies = data
            metrics_data[mode] = {
                'losses': all_losses,
                'accuracies': all_accuracies
            }
    
    for mode, data in EvalGraphs.items():
        if data:
            tp, fp, tn, fn, auc = data
            metrics_data[mode] = {
                'losses': metrics_data[mode]['losses'],
                'accuracies': metrics_data[mode]['accuracies'],
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
                'auc': auc
            }
    
    try:
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=4)
        Logging.info(f"正常：评估指标已保存至 {metrics_file}")
    except Exception as e:
        Logging.info(f"错误：保存评估指标时发生错误 - {str(e)}")

# 分割数据集，复制文件并创建测试集
def SplitApartData(paraDict):
    input_file = paraDict['FileIn'] + ".fasta"
    output_dir = os.path.join('backend/Output', paraDict['UserName'], paraDict['FileOut'])
    # 复制原始文件到 Output 文件夹
    import shutil
    shutil.copy(input_file, os.path.join(output_dir, os.path.basename(input_file)))
    
    # 读取输入文件内容
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 按序列分割数据
    sequences = []
    i = 0
    while i < len(lines):
        if lines[i].startswith('>'):
            header = lines[i].strip()
            seq = lines[i + 1].strip()
            sequences.append((header, seq))
            i += 2
        else:
            i += 1
    
    # 随机打乱序列
    import random
    random.shuffle(sequences)
    
    # 计算测试集数量
    test_ratio = float(paraDict['Test'])
    test_size = int(len(sequences) * test_ratio)
    
    # 分割训练集和测试集
    test_sequences = sequences[:test_size]
    train_sequences = sequences[test_size:]
    
    # 写入测试集文件
    test_file_path = os.path.join(output_dir, "Input-Test.fasta")
    with open(test_file_path, 'w', encoding='utf-8') as f:
        for header, seq in test_sequences:
            f.write(header + '\n')
            f.write(seq + '\n')
    
    # 写入更新后的训练集文件
    train_file_path = os.path.join(output_dir, "Input-Train.fasta")
    with open(train_file_path, 'w', encoding='utf-8') as f:
        for header, seq in train_sequences:
            f.write(header + '\n')
            f.write(seq + '\n')
    
    Logging.info(f"正常：已将 {test_ratio * 100}% 的数据分割到 {test_file_path}")
    Logging.info(f"正常：训练集已保存到 {train_file_path}")

# 读取命令行参数
def Dict():
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
        'Test':0.2,
        'AADict':"ACDEFGHIKLMNPQRSTVWY",
        'VoidDict':"X",
        #学习率
        'LearningRate': 0.001,
        #模型选用
        'Mode':"cnn",
        #训练分组数量
        'BatchSize': 32,
        #编码方式
        'EncodeMode':"one-hot",
        #训练轮数
        'Epochs': 10,
        #输入输出文件
        'FileIn': None,
        'FileOut': f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
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

    if paraDict['FileIn'] + ".fasta" is None:
        Logging.info("错误：未设置训练文件")
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


    if not os.path.exists('backend/Output/' + paraDict['UserName'] + '/' + paraDict['FileOut']):
        os.makedirs('backend/Output/' + paraDict['UserName'] + '/' + paraDict['FileOut'])

    return paraDict

# 新建参数包
def NewDict(
        UserName,
        FileIn,
        MaxLength = 10,
        Test = 0.2,
        AADict = "ACDEFGHIKLMNPQRSTVWY",
        VoidDict = "X",
        LearningRate = 0.001,
        Mode = "cnn",
        BatchSize = 32,
        EncodeMode = "one-hot",
        Epochs = 10
    ):
    paraDict = {
        'UserName': UserName,
        'MaxLength':MaxLength,
        'Test':Test,
        'AADict':AADict,
        'VoidDict':VoidDict,
        'LearningRate': LearningRate,
        'Mode':Mode,
        'BatchSize': BatchSize,
        'EncodeMode':EncodeMode,
        'Epochs': Epochs,
        'FileIn': FileIn,
        'FileOut': f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    }

    return paraDict

# 训练
def ModesTrain(paraDict):
    
    modes = paraDict['Mode'].split('+')
    SplitApartData(paraDict)

    all_sequences, all_labels = sample(paraDict, "backend/Output/" + paraDict['UserName'] + "/" + paraDict['FileOut'] + "/" + "Input-Train.fasta")
    Logging.info("正常：已合并训练正负样本")
    Config(paraDict)

    TrainGraphs = {
        'cnn': "",
        'rnn': "",
        'mlp': "",
        'lstm': "",
        'gcn': "",
        'gnn': "",
        'transformer': ""
    }

    if "cnn" in modes:
        from mode.cnn.cnn import cnn_train
        TrainGraphs['cnn'] = cnn_train(paraDict, all_sequences, all_labels, Logging, TimeStart)
    if "rnn" in modes:
        from mode.rnn.rnn import rnn_train
        TrainGraphs['rnn'] = rnn_train(paraDict, all_sequences, all_labels, Logging, TimeStart)
    if "mlp" in modes:
        from mode.mlp.mlp import mlp_train
        TrainGraphs['mlp'] = mlp_train(paraDict, all_sequences, all_labels, Logging, TimeStart)
    if "lstm" in modes:
        from mode.lstm.lstm import lstm_train
        TrainGraphs['lstm'] = lstm_train(paraDict, all_sequences, all_labels, Logging, TimeStart)
    if "transformer" in modes:
        from mode.transformer.transformer import transformer_train
        TrainGraphs['transformer'] = transformer_train(paraDict, all_sequences, all_labels, Logging, TimeStart)
    
    plot_training_progress(paraDict, TrainGraphs)
    return TrainGraphs

# 测试
def ModesTest(paraDict, TrainGraphs):

    modes = paraDict['Mode'].split('+')

    all_sequences, all_labels = sample(paraDict, "backend/Output/" + paraDict['UserName'] + "/" + paraDict['FileOut'] + "/" + "Input-Test.fasta")
    Logging.info("正常：已合并测试正负样本")

    EvalGraphs = {
        'cnn': "",
        'rnn': "",
        'mlp': "",
        'lstm': "",
        'gcn': "",
        'gnn': "",
        'transformer': ""
    }

    if "cnn" in modes:
        from mode.cnn.cnn import cnn_eval
        EvalGraphs['cnn'] = cnn_eval(paraDict, all_sequences, all_labels, Logging, TimeStart)
    if "rnn" in modes:
        from mode.rnn.rnn import rnn_eval
        EvalGraphs['rnn'] = rnn_eval(paraDict, all_sequences, all_labels, Logging, TimeStart)
    if "mlp" in modes:
        from mode.mlp.mlp import mlp_eval
        EvalGraphs['mlp'] = mlp_eval(paraDict, all_sequences, all_labels, Logging, TimeStart)
    if "lstm" in modes:
        from mode.lstm.lstm import lstm_eval
        EvalGraphs['lstm'] = lstm_eval(paraDict, all_sequences, all_labels, Logging, TimeStart)
    if "transformer" in modes:
        from mode.transformer.transformer import transformer_eval
        EvalGraphs['transformer'] = transformer_eval(paraDict, all_sequences, all_labels, Logging, TimeStart)

    cmc(paraDict, TrainGraphs, EvalGraphs)


if __name__ == '__main__':
    TimeStart = time.time()
    Logging = logger()
    Logging.info("================程序开始运行================")
    paraDict = Dict()
    TrainGraphs = ModesTrain(paraDict)
    ModesTest(paraDict, TrainGraphs)
    Logging.info("================程序结束运行================")
    Logging.info(f"总运行时间: {time.time() - TimeStart:.4f} 秒")
