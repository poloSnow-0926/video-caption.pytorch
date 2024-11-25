import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import pandas as pd
import re
import os
import numpy as np

def extract_video_features(video_path, model):
    """提取视频特征"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame).unsqueeze(0)
        frames.append(frame)
    
    cap.release()
    
    # 将所有帧堆叠成一个批次
    frames = torch.cat(frames, dim=0)
    
    # 提取特征
    with torch.no_grad():
        features = model(frames)
    
    # 返回平均特征
    return features.mean(dim=0).numpy()

def process_videos(video_dir, output_dir, csv_path):
    """处理所有视频并保存特征"""
    # 加载预训练模型
    model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])  # 移除最后的分类层
    model.eval()
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 创建特征字典
    features_dict = {}
    
    # 获取视频文件列表
    video_files = os.listdir(video_dir)
    
    for video_file in video_files:
        # 使用正则表达式提取video_id
        match = re.match(r"(G_\d+)", video_file)
        if match:
            video_id = match.group(1)
            if video_id in df['video_id'].values:
                video_path = os.path.join(video_dir, video_file)
                features = extract_video_features(video_path, model)
                features_dict[video_id] = features
    
    # 保存特征
    np.save(os.path.join(output_dir, 'video_features.npy'), features_dict)
    return features_dict
