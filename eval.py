import json
import os
import argparse
import torch
from torch.utils.data import DataLoader
from models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel
from dataloader import VideoDataset
import misc.utils as utils
import pandas as pd
from tqdm import tqdm

def generate_captions(model, dataset, vocab, opt):
    """生成视频描述并保存为CSV"""
    model.eval()
    loader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=False)
    
    # 用于存储结果的字典
    results = {}
    
    # 使用tqdm显示进度
    for data in tqdm(loader, desc="Generating captions"):
        # 将特征移至GPU
        fc_feats = data['fc_feats'].cuda()
        video_ids = data['video_ids']
        
        # 生成预测
        with torch.no_grad():
            seq_probs, seq_preds = model(
                fc_feats, mode='inference', opt=opt)
        
        # 解码序列得到描述文本
        sents = utils.decode_sequence(vocab, seq_preds)
        
        # 保存结果
        for vid, sent in zip(video_ids, sents):
            results[vid] = sent
    
    # 读取原始test.csv
    df = pd.read_csv(opt["test_csv"])
    
    # 更新caption列
    df['caption'] = df['video_id'].map(results)
    
    # 保存结果
    output_path = os.path.join(opt["results_path"], "submission.csv")
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def main(opt):
    # 加载数据集
    dataset = VideoDataset(opt, "test")
    opt["vocab_size"] = dataset.get_vocab_size()
    opt["seq_length"] = dataset.max_len
    
    # 初始化模型
    if opt["model"] == 'S2VTModel':
        model = S2VTModel(
            opt["vocab_size"], 
            opt["max_len"], 
            opt["dim_hidden"], 
            opt["dim_word"],
            rnn_dropout_p=opt["rnn_dropout_p"]
        ).cuda()
    elif opt["model"] == "S2VTAttModel":
        encoder = EncoderRNN(
            opt["dim_vid"], 
            opt["dim_hidden"],
            bidirectional=opt["bidirectional"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_dropout_p=opt["rnn_dropout_p"]
        )
        decoder = DecoderRNN(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_dropout_p=opt["rnn_dropout_p"],
            bidirectional=opt["bidirectional"]
        )
        model = S2VTAttModel(encoder, decoder).cuda()
    
    # 加载模型权重
    model.load_state_dict(torch.load(opt["saved_model"]))
    print(f"Loaded model from {opt['saved_model']}")
    
    # 生成描述并保存CSV
    generate_captions(model, dataset, dataset.get_vocab(), opt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recover_opt', type=str, required=True,
                        help='recover train opts from saved opt_json')
    parser.add_argument('--saved_model', type=str, required=True,
                        help='path to saved model to evaluate')
    parser.add_argument('--test_csv', type=str, required=True,
                        help='path to test.csv file')
    parser.add_argument('--results_path', type=str, default='results/')
    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu device number')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--sample_max', type=int, default=1,
                        help='0/1. whether sample max probs to get next word in inference stage')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--beam_size', type=int, default=1,
                        help='used when sample_max = 1. Usually 2 or 3 works well.')

    args = parser.parse_args()
    args = vars((args))
    opt = json.load(open(args["recover_opt"]))
    for k, v in args.items():
        opt[k] = v
    
    main(opt)
