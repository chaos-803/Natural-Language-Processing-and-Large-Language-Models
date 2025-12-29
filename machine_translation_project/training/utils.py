"""
训练工具函数
包含训练循环、验证、早停、学习率调度等功能
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import numpy as np
import time
import os
from typing import Tuple, Dict, Any, List, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0, verbose: bool = False):
        """
        参数:
            patience: 容忍的验证损失不下降的轮数
            min_delta: 最小改善阈值
            verbose: 是否打印早停信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None
    
    def __call__(self, val_loss: float, model: nn.Module):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'早停计数器: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state = model.state_dict().copy()
            self.counter = 0
    
    def should_stop(self) -> bool:
        return self.early_stop
    
    def get_best_state(self):
        return self.best_state


class WarmupCosineSchedule(LambdaLR):
    """带预热的学习率调度器"""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, 
                 min_lr: float = 1e-7, last_epoch: int = -1):
        """
        参数:
            optimizer: 优化器
            warmup_steps: 预热步数
            total_steps: 总训练步数
            min_lr: 最小学习率
            last_epoch: 最后一个epoch的索引
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                # 线性预热
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # 余弦衰减
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(min_lr, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        super().__init__(optimizer, lr_lambda, last_epoch)


class Trainer:
    """模型训练器"""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], 
                 device: torch.device, checkpoint_dir: str = 'checkpoints'):
        """
        参数:
            model: 要训练的模型
            config: 训练配置
            device: 训练设备
            checkpoint_dir: 检查点保存目录
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 损失函数（忽略填充标记）
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config.get('learning_rate', 0.001),
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # 学习率调度器
        if config.get('use_warmup', False):
            self.scheduler = WarmupCosineSchedule(
                self.optimizer,
                warmup_steps=config.get('warmup_steps', 4000),
                total_steps=config.get('total_steps', 100000)
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=5, 
                verbose=True
            )
        
        # 早停机制
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 10),
            min_delta=config.get('min_delta', 0.001),
            verbose=True
        )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_ppl': [],
            'val_ppl': [],
            'learning_rates': []
        }
        
        # 初始化最佳模型状态
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        print(f"训练器初始化完成，使用设备: {device}")
    
    def train_epoch(self, train_loader, epoch: int) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} [训练]')
        
        for batch_idx, batch in enumerate(progress_bar):
            # 准备数据
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            src_lengths = batch['src_lengths'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            # 根据模型类型调整输入
            if hasattr(self.model, 'forward'):
                # RNN模型
                output = self.model(src, src_lengths, tgt[:, :-1])
                loss = self.criterion(
                    output.contiguous().view(-1, output.size(-1)),
                    tgt[:, 1:].contiguous().view(-1)
                )
            else:
                # Transformer模型
                output = self.model(src, tgt[:, :-1])
                loss = self.criterion(
                    output.contiguous().view(-1, output.size(-1)),
                    tgt[:, 1:].contiguous().view(-1)
                )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 优化步骤
            self.optimizer.step()
            
            # 更新学习率（如果使用预热调度器）
            if isinstance(self.scheduler, WarmupCosineSchedule):
                self.scheduler.step()
            
            # 统计
            batch_loss = loss.item()
            batch_tokens = (tgt[:, 1:] != 0).sum().item()
            
            total_loss += batch_loss * batch_tokens
            total_tokens += batch_tokens
            
            # 更新进度条
            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
            progress_bar.set_postfix({'loss': f'{batch_loss:.4f}', 'avg_loss': f'{avg_loss:.4f}'})
            
            # 记录学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            if batch_idx % 100 == 0:
                self.history['learning_rates'].append(current_lr)
        
        # 计算平均损失和困惑度
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        ppl = np.exp(avg_loss) if avg_loss < 100 else float('inf')
        
        return avg_loss, ppl
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc='验证')
            
            for batch in progress_bar:
                # 准备数据
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                src_lengths = batch['src_lengths'].to(self.device)
                
                # 前向传播
                if hasattr(self.model, 'forward'):
                    # RNN模型
                    output = self.model(src, src_lengths, tgt[:, :-1])
                else:
                    # Transformer模型
                    output = self.model(src, tgt[:, :-1])
                
                # 计算损失
                loss = self.criterion(
                    output.contiguous().view(-1, output.size(-1)),
                    tgt[:, 1:].contiguous().view(-1)
                )
                
                # 统计
                batch_loss = loss.item()
                batch_tokens = (tgt[:, 1:] != 0).sum().item()
                
                total_loss += batch_loss * batch_tokens
                total_tokens += batch_tokens
                
                # 更新进度条
                avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
                progress_bar.set_postfix({'loss': f'{batch_loss:.4f}', 'avg_loss': f'{avg_loss:.4f}'})
        
        # 计算平均损失和困惑度
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        ppl = np.exp(avg_loss) if avg_loss < 100 else float('inf')
        
        return avg_loss, ppl
    
    def train(self, train_loader, val_loader, num_epochs: int):
        """完整训练流程"""
        print(f"开始训练，共 {num_epochs} 个epoch")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"可训练参数数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 训练一个epoch
            train_loss, train_ppl = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss, val_ppl = self.validate(val_loader)
            
            # 更新学习率（如果使用ReduceLROnPlateau）
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_ppl'].append(train_ppl)
            self.history['val_ppl'].append(val_ppl)
            
            # 打印epoch结果
            epoch_time = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f'\nEpoch {epoch+1}/{num_epochs}:')
            print(f'  时间: {epoch_time:.2f}s')
            print(f'  学习率: {current_lr:.6f}')
            print(f'  训练损失: {train_loss:.4f} | 训练困惑度: {train_ppl:.2f}')
            print(f'  验证损失: {val_loss:.4f} | 验证困惑度: {val_ppl:.2f}')
            
            # 检查是否是最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                
                # 保存最佳模型
                self.save_checkpoint(f'best_model_epoch{epoch+1}.pth', epoch, val_loss)
                print(f'  ✓ 保存最佳模型，验证损失: {val_loss:.4f}')
            
            # 定期保存检查点
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch{epoch+1}.pth', epoch, val_loss)
            
            # 早停检查
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.should_stop():
                print(f'早停触发，最佳验证损失: {self.early_stopping.best_loss:.4f}')
                break
        
        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f'恢复最佳模型，验证损失: {self.best_val_loss:.4f}')
        
        print('训练完成!')
        return self.history
    
    def save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'history': self.history,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        
        # 同时保存为最新检查点
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        print(f"加载检查点: epoch {checkpoint['epoch']}, 验证损失: {checkpoint['val_loss']:.4f}")
    
    def plot_training_history(self, save_path: str = None):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='训练损失')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='验证损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].set_title('训练和验证损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 困惑度曲线
        axes[0, 1].plot(epochs, self.history['train_ppl'], 'b-', label='训练困惑度')
        axes[0, 1].plot(epochs, self.history['val_ppl'], 'r-', label='验证困惑度')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('困惑度')
        axes[0, 1].set_title('训练和验证困惑度')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].set_yscale('log')
        
        # 学习率曲线
        if self.history['learning_rates']:
            lr_steps = range(len(self.history['learning_rates']))
            axes[1, 0].plot(lr_steps, self.history['learning_rates'], 'g-')
            axes[1, 0].set_xlabel('训练步数')
            axes[1, 0].set_ylabel('学习率')
            axes[1, 0].set_title('学习率变化')
            axes[1, 0].grid(True)
        
        # 损失对比（对数坐标）
        axes[1, 1].plot(epochs, self.history['train_loss'], 'b-', label='训练损失')
        axes[1, 1].plot(epochs, self.history['val_loss'], 'r-', label='验证损失')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('损失（对数）')
        axes[1, 1].set_title('损失对比（对数坐标）')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存到: {save_path}")
        
        plt.show()
    
    def save_training_history(self, filepath: str):
        """保存训练历史到JSON文件"""
        history_dict = {
            'train_loss': [float(x) for x in self.history['train_loss']],
            'val_loss': [float(x) for x in self.history['val_loss']],
            'train_ppl': [float(x) for x in self.history['train_ppl']],
            'val_ppl': [float(x) for x in self.history['val_ppl']],
            'learning_rates': [float(x) for x in self.history['learning_rates']],
            'best_val_loss': float(self.best_val_loss)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history_dict, f, indent=2, ensure_ascii=False)
        
        print(f"训练历史已保存到: {filepath}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def initialize_weights(model: nn.Module):
    """初始化模型权重"""
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


# 测试代码
if __name__ == "__main__":
    # 测试Trainer类
    print("测试训练器...")
    
    # 创建虚拟模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 128)
            self.linear = nn.Linear(128, 1000)
        
        def forward(self, x, y):
            return self.linear(self.embedding(x)).mean(dim=1).unsqueeze(1).expand(-1, y.size(1)-1, -1)
    
    # 创建虚拟数据加载器
    class TestDataLoader:
        def __init__(self):
            self.batches = []
            for _ in range(10):
                batch = {
                    'src': torch.randint(1, 100, (16, 20)),
                    'tgt': torch.randint(1, 100, (16, 25)),
                    'src_lengths': torch.randint(10, 20, (16,))
                }
                self.batches.append(batch)
        
        def __iter__(self):
            return iter(self.batches)
        
        def __len__(self):
            return len(self.batches)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型和训练器
    model = TestModel()
    config = {
        'learning_rate': 0.001,
        'use_warmup': False,
        'patience': 3,
        'min_delta': 0.001
    }
    
    trainer = Trainer(model, config, device, checkpoint_dir='test_checkpoints')
    
    # 测试训练一个epoch
    print("\n测试训练一个epoch...")
    train_loader = TestDataLoader()
    val_loader = TestDataLoader()
    
    train_loss, train_ppl = trainer.train_epoch(train_loader, 0)
    print(f"训练损失: {train_loss:.4f}, 训练困惑度: {train_ppl:.4f}")
    
    # 测试验证
    print("\n测试验证...")
    val_loss, val_ppl = trainer.validate(val_loader)
    print(f"验证损失: {val_loss:.4f}, 验证困惑度: {val_ppl:.4f}")
    
    # 测试参数统计
    print("\n测试参数统计...")
    param_counts = count_parameters(model)
    print(f"总参数: {param_counts['total']:,}")
    print(f"可训练参数: {param_counts['trainable']:,}")
    print(f"不可训练参数: {param_counts['non_trainable']:,}")
    
    # 测试权重初始化
    print("\n测试权重初始化...")
    initialize_weights(model)
    print("权重初始化完成")
    
    # 测试检查点保存和加载
    print("\n测试检查点保存和加载...")
    trainer.save_checkpoint('test_checkpoint.pth', 0, val_loss)
    print("检查点保存完成")
    
    # 清理测试文件
    import os
    import shutil
    if os.path.exists('test_checkpoints'):
        shutil.rmtree('test_checkpoints')
    
    print("\n所有测试完成!")