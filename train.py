import json
import os
import time
from tqdm import tqdm
import numpy as np

import misc.utils as utils
import opts
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from dataloader import VideoDataset
from misc.rewards import get_self_critical_reward, init_cider_scorer
from models import DecoderRNN, EncoderRNN, S2VTAttModel, S2VTModel
from torch import nn
from torch.utils.data import DataLoader

def train(loader, model, crit, optimizer, lr_scheduler, opt, rl_crit=None):
    model.train()
    
    # 添加日志记录
    log_path = os.path.join(opt["checkpoint_path"], 'training_log.txt')
    
    # 添加最佳模型保存
    best_loss = float('inf')
    best_model_path = os.path.join(opt["checkpoint_path"], 'model_best.pth')
    
    for epoch in range(opt["epochs"]):
        epoch_start_time = time.time()
        total_loss = 0
        iteration = 0
        
        # 设置自监督训练标志
        if opt["self_crit_after"] != -1 and epoch >= opt["self_crit_after"]:
            sc_flag = True
            init_cider_scorer(opt["cached_tokens"])
        else:
            sc_flag = False

        # 使用tqdm显示进度条
        progress = tqdm(loader, desc=f'Epoch {epoch}')
        for data in progress:
            torch.cuda.synchronize()
            
            # 数据移至GPU
            fc_feats = data['fc_feats'].cuda()
            labels = data['labels'].cuda()
            masks = data['masks'].cuda()

            optimizer.zero_grad()
            
            if not sc_flag:
                # 普通训练模式
                seq_probs, _ = model(fc_feats, labels, 'train')
                loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])
            else:
                # 自监督训练模式
                seq_probs, seq_preds = model(fc_feats, mode='inference', opt=opt)
                reward = get_self_critical_reward(model, fc_feats, data, seq_preds)
                loss = rl_crit(seq_probs, seq_preds,
                             torch.from_numpy(reward).float().cuda())

            loss.backward()
            
            # 梯度裁剪
            clip_grad_value_(model.parameters(), opt['grad_clip'])
            optimizer.step()
            
            train_loss = loss.item()
            total_loss += train_loss
            
            # 更新进度条
            progress.set_postfix({'loss': f'{train_loss:.4f}'})
            
            torch.cuda.synchronize()
            iteration += 1

        # 计算平均损失
        avg_loss = total_loss / iteration
        epoch_time = time.time() - epoch_start_time
        
        # 记录训练日志
        log_msg = f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s"
        with open(log_path, 'a') as f:
            f.write(log_msg + '\n')
        print(log_msg)

        # 保存检查点
        if epoch % opt["save_checkpoint_every"] == 0:
            model_path = os.path.join(opt["checkpoint_path"],
                                    f'model_{epoch}.pth')
            model_info_path = os.path.join(opt["checkpoint_path"],
                                         'model_score.txt')
            
            # 保存模型状态
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(state, model_path)
            
            # 记录模型信息
            with open(model_info_path, 'a') as f:
                f.write(f"model_{epoch}, loss: {avg_loss:.6f}\n")
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(state, best_model_path)
                print(f"New best model saved! Loss: {best_loss:.6f}")
        
        # 学习率调整
        lr_scheduler.step()

def main(opt):
    # 设置随机种子
    torch.manual_seed(opt.get('seed', 42))
    np.random.seed(opt.get('seed', 42))
    
    # 创建数据加载器
    dataset = VideoDataset(opt, 'train')
    dataloader = DataLoader(
        dataset, 
        batch_size=opt["batch_size"], 
        shuffle=True,
        num_workers=opt.get('num_workers', 4),
        pin_memory=True
    )
    
    opt["vocab_size"] = dataset.get_vocab_size()
    
    # 模型选择和初始化
    if opt["model"] == 'S2VTModel':
        model = S2VTModel(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            opt['dim_vid'],
            rnn_cell=opt['rnn_type'],
            n_layers=opt['num_layers'],
            rnn_dropout_p=opt["rnn_dropout_p"]
        )
    elif opt["model"] == "S2VTAttModel":
        encoder = EncoderRNN(
            opt["dim_vid"],
            opt["dim_hidden"],
            bidirectional=opt["bidirectional"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"]
        )
        decoder = DecoderRNN(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"],
            bidirectional=opt["bidirectional"]
        )
        model = S2VTAttModel(encoder, decoder)
    
    # 移至GPU
    model = model.cuda()
    
    # 定义损失函数
    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()
    
    # 优化器设置
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt["learning_rate"],
        weight_decay=opt["weight_decay"],
        betas=(opt["optim_alpha"], opt["optim_beta"]),
        eps=opt["optim_epsilon"]
    )
    
    # 学习率调度器
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt["learning_rate_decay_every"],
        gamma=opt["learning_rate_decay_rate"]
    )

    # 开始训练
    train(dataloader, model, crit, optimizer, exp_lr_scheduler, opt, rl_crit)

if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    
    # GPU设置
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    
    # 创建检查点目录
    if not os.path.isdir(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    
    # 保存配置
    opt_json = os.path.join(opt["checkpoint_path"], 'opt_info.json')
    with open(opt_json, 'w') as f:
        json.dump(opt, f, indent=4)
    print('save opt details to %s' % (opt_json))
    
    # 开始训练
    main(opt)
