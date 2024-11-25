import pandas as pd
import ast
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import re
import random

class CustomVideoDataset(Dataset):
    def __init__(self, opt, mode='train'):
        super(CustomVideoDataset, self).__init__()
        self.mode = mode
        self.max_len = opt["max_len"]
        
        # 读取CSV数据
        self.df = pd.read_csv(opt["caption_csv"])
        # 转换captions从字符串到列表
        self.df['captions'] = self.df['captions'].apply(ast.literal_eval)
        
        # 构建词汇表
        self.build_vocab()
        
        # 特征目录
        self.feats_dir = opt["feats_dir"]
        self.with_c3d = opt.get('with_c3d', 0)
        self.c3d_feats_dir = opt.get('c3d_feats_dir', '')
        
        # 获取视频文件列表并建立映射
        self.video_files = self.get_video_files()
        
    def build_vocab(self):
        # 构建词汇表
        word_counts = {}
        for captions in self.df['captions']:
            for caption in captions:
                for word in caption.split():
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # 添加特殊标记
        self.word_to_ix = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        for word in word_counts:
            if word not in self.word_to_ix:
                self.word_to_ix[word] = len(self.word_to_ix)
        
        self.ix_to_word = {v: k for k, v in self.word_to_ix.items()}
        
    def get_video_files(self):
        # 获取特征文件列表并建立与video_id的映射
        video_mapping = {}
        for filename in os.listdir(self.feats_dir):
            if filename.endswith('.npy'):
                # 使用正则表达式提取video_id
                match = re.match(r"(G_\d+)", filename)
                if match:
                    video_id = match.group(1)
                    video_mapping[video_id] = filename
        return video_mapping
    
    def get_vocab_size(self):
        return len(self.word_to_ix)
    
    def get_vocab(self):
        return self.ix_to_word
    
    def process_caption(self, caption):
        # 将caption转换为词索引序列
        words = caption.split()
        if len(words) > self.max_len - 2:  # 留出<sos>和<eos>的位置
            words = words[:(self.max_len - 2)]
        
        indices = [self.word_to_ix.get(word, self.word_to_ix['<unk>']) for word in words]
        indices = [self.word_to_ix['<sos>']] + indices + [self.word_to_ix['<eos>']]
        
        return indices
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = row['video_id']
        captions = row['captions']
        
        # 加载特征
        feat_path = os.path.join(self.feats_dir, self.video_files[video_id])
        fc_feat = np.load(feat_path)
        
        # 处理C3D特征（如果需要）
        if self.with_c3d:
            c3d_feat = np.load(os.path.join(self.c3d_feats_dir, self.video_files[video_id]))
            c3d_feat = np.mean(c3d_feat, axis=0, keepdims=True)
            fc_feat = np.concatenate((fc_feat, np.tile(c3d_feat, (fc_feat.shape[0], 1))), axis=1)
        
        # 处理所有caption
        processed_captions = [self.process_caption(cap) for cap in captions]
        
        # 填充到相同长度
        gts = np.zeros((len(processed_captions), self.max_len))
        for i, cap in enumerate(processed_captions):
            gts[i, :len(cap)] = cap
        
        # 随机选择一个caption
        cap_ix = random.randint(0, len(processed_captions) - 1)
        label = gts[cap_ix]
        
        # 创建mask
        mask = np.zeros(self.max_len)
        non_zero = (label == 0).nonzero()
        mask[:int(non_zero[0][0]) + 1] = 1
        
        # 返回数据字典
        data = {
            'fc_feats': torch.from_numpy(fc_feat).type(torch.FloatTensor),
            'labels': torch.from_numpy(label).type(torch.LongTensor),
            'masks': torch.from_numpy(mask).type(torch.FloatTensor),
            'gts': torch.from_numpy(gts).long(),
            'video_ids': video_id
        }
        
        return data
    
    def __len__(self):
        return len(self.df)
