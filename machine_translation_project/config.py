import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = "data"
    train_small_path: str = "data/train_small.jsonl"
    train_large_path: str = "data/train_large.jsonl" 
    valid_path: str = "data/valid.jsonl"
    test_path: str = "data/test.jsonl"
    
    # 预处理配置
    max_seq_len: int = 100
    min_freq: int = 2
    src_lang: str = "zh"
    tgt_lang: str = "en"
    
    # 分词配置
    tokenizer_type: str = "bpe"  # bpe, word, char
    vocab_size: int = 30000

@dataclass
class RNNConfig:
    """RNN模型配置"""
    # 模型架构
    hidden_size: int = 512
    num_layers: int = 2
    embedding_dim: int = 512
    dropout: float = 0.3
    bidirectional: bool = True
    cell_type: str = "lstm"  # lstm, gru
    
    # 注意力机制
    attention_method: str = "dot"  # dot, general, concat
    attention_size: int = 512
    
    # 训练配置
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 50
    teacher_forcing_ratio: float = 0.5
    max_grad_norm: float = 5.0
    
    # 解码配置
    beam_size: int = 5
    max_decode_len: int = 100

@dataclass
class TransformerConfig:
    """Transformer模型配置"""
    # 模型架构
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    
    # 位置编码
    pos_encoding: str = "absolute"  # absolute, relative
    
    # 训练配置  
    batch_size: int = 64
    learning_rate: float = 0.0001
    num_epochs: int = 50
    warmup_steps: int = 4000
    label_smoothing: float = 0.1
    
    # 解码配置
    beam_size: int = 5
    max_decode_len: int = 100

@dataclass
class TrainingConfig:
    """训练配置"""
    device: str = "cuda"  # cuda, cpu
    seed: int = 42
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    result_dir: str = "results"
    
    # 早停配置
    patience: int = 10
    min_delta: float = 0.001
    
    # 日志配置
    log_interval: int = 100
    eval_interval: int = 1000

class Config:
    """总配置类"""
    def __init__(self):
        self.data = DataConfig()
        self.rnn = RNNConfig()
        self.transformer = TransformerConfig()
        self.training = TrainingConfig()
        
        # 设置设备
        if self.training.device == "cuda" and not torch.cuda.is_available():
            self.training.device = "cpu"
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "data": self.data.__dict__,
            "rnn": self.rnn.__dict__, 
            "transformer": self.transformer.__dict__,
            "training": self.training.__dict__
        }
    
    def save(self, path: str):
        """保存配置"""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def load(self, path: str):
        """加载配置"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for key, value in data.items():
            if hasattr(self, key):
                config_obj = getattr(self, key)
                for k, v in value.items():
                    if hasattr(config_obj, k):
                        setattr(config_obj, k, v)

# 全局配置实例
config = Config()

# 导入torch（在类定义后导入以避免循环依赖）
import torch