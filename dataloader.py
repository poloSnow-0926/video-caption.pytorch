import pandas as pd
import ast
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import re
import random
import cv2
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self):
        # 加载预训练的ResNet模型
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # 移除最后的分类层
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, video_path):
        """从视频中提取特征"""
        cap = cv2.VideoCapture(video_path)
        features = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 预处理帧
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame).unsqueeze(0)
            
            if torch.cuda.is_available():
                frame = frame.cuda()
            
            # 提取特征
            with torch.no_grad():
                feat = self.model(frame)
                features.append(feat.cpu().numpy().squeeze())
        
        cap.release()
        return np.array(features)

class CustomVideoDataset(Dataset):
    def __init__(self, opt, mode='train'):
        super(CustomVideoDataset, self).__init__()
        self.mode = mode
        self.max_len = opt["max_len"]
        self.feature_dim = opt.get("feature_dim", 2048)  # ResNet特征维度
        
        # 读取CSV数据
        self.df = pd.read_csv(opt["caption_csv"])
        self.df['captions'] = self.df['captions'].apply(ast.literal_eval)
        
        # 特征相关目录
        self.videos_dir = opt.get("videos_dir", "")  # 原始视频目录
        self.feats_dir = opt["feats_dir"]  # 特征保存目录
        self.with_c3d = opt.get('with_c3d', 0)
        self.c3d_feats_dir = opt.get('c3d_feats_dir', '')
        
        # 确保特征目录存在
        os.makedirs(self.feats_dir, exist_ok=True)
        
        # 构建词汇表
        self.build_vocab()
        
        # 提取或加载特征
        self.extract_and_save_features()
        
        # 获取视频文件列表并建立映射
        self.video_files = self.get_video_files()
        
    def build_vocab(self):
        """构建词汇表"""
        word_counts = {}
        for captions in self.df['captions']:
            for caption in captions:
                for word in caption.split():
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # 添加特殊标记
        self.word_to_ix = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        
        # 只包含出现次数大于阈值的词
        min_word_count = 5
        for word, count in word_counts.items():
            if count >= min_word_count and word not in self.word_to_ix:
                self.word_to_ix[word] = len(self.word_to_ix)
        
        self.ix_to_word = {v: k for k, v in self.word_to_ix.items()}
        
    def extract_and_save_features(self):
        """提取并保存视频特征"""
        if not os.path.exists(self.videos_dir):
            raise ValueError(f"Videos directory {self.videos_dir} does not exist!")
        
        # 初始化特征提取器
        feature_extractor = FeatureExtractor()
        
        # 获取所有视频文件
        video_files = [f for f in os.listdir(self.videos_dir) if f.endswith(('.mp4', '.avi'))]
        
        for video_file in tqdm(video_files, desc="Extracting features"):
            # 获取video_id
            match = re.match(r"(G_\d+)", video_file)
            if not match:
                continue
                
            video_id = match.group(1)
            feature_path = os.path.join(self.feats_dir, f"{video_id}_features.npy")
            
            # 如果特征文件不存在，则提取并保存
            if not os.path.exists(feature_path):
                video_path = os.path.join(self.videos_dir, video_file)
                features = feature_extractor.extract_features(video_path)
                np.save(feature_path, features)
    
    def get_video_files(self):
        """获取特征文件列表并建立与video_id的映射"""
        video_mapping = {}
        for filename in os.listdir(self.feats_dir):
            if filename.endswith('.npy'):
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
        """处理单个caption"""
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
        feat_path = os.path.join(self.feats_dir, f"{video_id}_features.npy")
        fc_feat = np.load(feat_path)
        
        # 处理C3D特征（如果需要）
        if self.with_c3d and self.c3d_feats_dir:
            c3d_feat_path = os.path.join(self.c3d_feats_dir, f"{video_id}_c3d.npy")
            if os.path.exists(c3d_feat_path):
                c3d_feat = np.load(c3d_feat_path)
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
        if len(non_zero[0]) > 0:
            mask[:int(non_zero[0][0]) + 1] = 1
        else:
            mask[:] = 1
        
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
