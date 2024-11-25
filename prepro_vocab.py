import pandas as pd
import ast
from collections import Counter
import json

def build_vocab(csv_path, min_freq=5):
    """构建词汇表"""
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    df['captions'] = df['captions'].apply(lambda x: ast.literal_eval(x))
    
    # 收集所有单词
    word_freq = Counter()
    for caption_list in df['captions']:
        for caption in caption_list:
            words = caption.lower().split()
            word_freq.update(words)
    
    # 创建词汇表
    vocab = {
        '<PAD>': 0,
        '<START>': 1,
        '<END>': 2,
        '<UNK>': 3,
    }
    
    # 添加频率大于min_freq的词
    idx = 4
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    
    # 保存词汇表
    return vocab

def save_vocab(vocab, output_path):
    """保存词汇表到文件"""
    with open(output_path, 'w') as f:
        json.dump(vocab, f)

# 主执行脚本
if __name__ == '__main__':
    # 设置路径
    VIDEO_DIR = 'path/to/your/videos'
    OUTPUT_DIR = 'path/to/output'
    CSV_PATH = 'train_data.csv'
    
    # 1. 提取视频特征
    features_dict = process_videos(VIDEO_DIR, OUTPUT_DIR, CSV_PATH)
    print(f"处理了 {len(features_dict)} 个视频")
    
    # 2. 构建词汇表
    vocab = build_vocab(CSV_PATH)
    save_vocab(vocab, os.path.join(OUTPUT_DIR, 'vocab.json'))
    print(f"词汇表大小: {len(vocab)}")
