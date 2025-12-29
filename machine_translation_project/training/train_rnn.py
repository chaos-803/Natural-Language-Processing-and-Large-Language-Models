"""
RNN神经机器翻译模型训练脚本
包含完整的训练流程、模型评估和结果保存
"""
import torch
import torch.nn as nn
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rnn_nmt import RNNNMT
from data.dataset import DataManager
from training.utils import Trainer, count_parameters, initialize_weights
from config import config

def setup_rnn_config(args):
    """设置RNN配置"""
    rnn_config = {
        # 模型架构
        'src_vocab_size': args.src_vocab_size,
        'tgt_vocab_size': args.tgt_vocab_size,
        'embedding_dim': args.embedding_dim,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'bidirectional': args.bidirectional,
        'cell_type': args.cell_type,
        'attention_method': args.attention_method,
        
        # 训练配置
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'teacher_forcing_ratio': args.teacher_forcing_ratio,
        'max_grad_norm': args.max_grad_norm,
        
        # 学习率调度
        'use_warmup': args.use_warmup,
        'warmup_steps': args.warmup_steps,
        'total_steps': args.total_steps,
        
        # 早停
        'patience': args.patience,
        'min_delta': args.min_delta,
        
        # 其他
        'device': args.device,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir
    }
    
    return rnn_config

def train_rnn_model(args):
    """训练RNN模型"""
    print("=" * 60)
    print("开始训练RNN神经机器翻译模型")
    print("=" * 60)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    data_manager = DataManager(config)
    train_loader, val_loader, test_loader = data_manager.load_data()
    vocab_info = data_manager.get_vocab_info()
    
    # 更新词表大小
    args.src_vocab_size = vocab_info['src_vocab_size']
    args.tgt_vocab_size = vocab_info['tgt_vocab_size']
    
    print(f"源语言词表大小: {args.src_vocab_size}")
    print(f"目标语言词表大小: {args.tgt_vocab_size}")
    
    # 创建模型
    print("\n[2/4] 创建RNN模型...")
    model_config = setup_rnn_config(args)
    
    model = RNNNMT(
        src_vocab_size=args.src_vocab_size,
        tgt_vocab_size=args.tgt_vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        cell_type=args.cell_type,
        attention_method=args.attention_method
    )
    
    # 初始化权重
    initialize_weights(model)
    
    # 统计参数
    param_counts = count_parameters(model)
    print(f"模型参数统计:")
    print(f"  总参数: {param_counts['total']:,}")
    print(f"  可训练参数: {param_counts['trainable']:,}")
    print(f"  不可训练参数: {param_counts['non_trainable']:,}")
    
    # 创建训练器
    print("\n[3/4] 创建训练器...")
    trainer = Trainer(model, model_config, device, args.checkpoint_dir)
    
    # 训练模型
    print("\n[4/4] 开始训练...")
    history = trainer.train(train_loader, val_loader, args.num_epochs)
    
    # 保存训练历史
    history_path = os.path.join(args.log_dir, 'rnn_training_history.json')
    trainer.save_training_history(history_path)
    
    # 绘制训练曲线
    plot_path = os.path.join(args.log_dir, 'rnn_training_plot.png')
    trainer.plot_training_history(plot_path)
    
    # 保存最终模型
    final_model_path = os.path.join(args.checkpoint_dir, 'rnn_final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_config,
        'vocab_info': vocab_info
    }, final_model_path)
    
    print(f"\n模型训练完成!")
    print(f"最终模型保存到: {final_model_path}")
    print(f"训练历史保存到: {history_path}")
    print(f"训练曲线保存到: {plot_path}")
    
    return model, history

def evaluate_rnn_model(model, test_loader, device):
    """评估RNN模型"""
    print("\n评估模型性能...")
    model.eval()
    
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    with torch.no_grad():
        for batch in test_loader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_lengths = batch['src_lengths'].to(device)
            
            # 前向传播
            output = model(src, src_lengths, tgt[:, :-1])
            
            # 计算损失
            loss = criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt[:, 1:].contiguous().view(-1)
            )
            
            # 统计
            batch_tokens = (tgt[:, 1:] != 0).sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    
    print(f"测试集损失: {avg_loss:.4f}")
    print(f"测试集困惑度: {ppl:.2f}")
    
    return avg_loss, ppl

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练RNN神经机器翻译模型')
    
    # 模型架构参数
    parser.add_argument('--embedding-dim', type=int, default=512, help='词嵌入维度')
    parser.add_argument('--hidden-size', type=int, default=512, help='隐藏层大小')
    parser.add_argument('--num-layers', type=int, default=2, help='RNN层数')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout率')
    parser.add_argument('--bidirectional', action='store_true', default=True, help='是否使用双向RNN')
    parser.add_argument('--cell-type', type=str, default='lstm', choices=['lstm', 'gru'], help='RNN单元类型')
    parser.add_argument('--attention-method', type=str, default='dot', 
                       choices=['dot', 'general', 'concat', 'bahdanau'], help='注意力机制类型')
    
    # 训练参数
    parser.add_argument('--learning-rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--batch-size', type=int, default=64, help='批大小')
    parser.add_argument('--num-epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--teacher-forcing-ratio', type=float, default=0.5, help='教师强制比率')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='梯度裁剪阈值')
    
    # 学习率调度参数
    parser.add_argument('--use-warmup', action='store_true', default=False, help='使用学习率预热')
    parser.add_argument('--warmup-steps', type=int, default=4000, help='预热步数')
    parser.add_argument('--total-steps', type=int, default=100000, help='总训练步数')
    
    # 早停参数
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--min-delta', type=float, default=0.001, help='最小改善阈值')
    
    # 数据参数
    parser.add_argument('--src-vocab-size', type=int, default=30000, help='源语言词表大小')
    parser.add_argument('--tgt-vocab-size', type=int, default=30000, help='目标语言词表大小')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/rnn', help='检查点保存目录')
    parser.add_argument('--log-dir', type=str, default='logs/rnn', help='日志保存目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 创建目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(args.log_dir, 'rnn_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f"配置保存到: {config_path}")
    
    # 训练模型
    model, history = train_rnn_model(args)
    
    # 评估模型
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_manager = DataManager(config)
    _, _, test_loader = data_manager.load_data()
    
    test_loss, test_ppl = evaluate_rnn_model(model, test_loader, device)
    
    # 保存评估结果
    eval_result = {
        'test_loss': test_loss,
        'test_ppl': test_ppl,
        'best_val_loss': min(history['val_loss']) if history['val_loss'] else None,
        'best_val_ppl': min(history['val_ppl']) if history['val_ppl'] else None,
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
        'final_train_ppl': history['train_ppl'][-1] if history['train_ppl'] else None,
        'training_time': len(history['train_loss'])  # 实际训练轮数
    }
    
    eval_path = os.path.join(args.log_dir, 'rnn_evaluation.json')
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n评估结果保存到: {eval_path}")
    print("RNN模型训练完成!")

if __name__ == "__main__":
    main()