"""
数据加载器和数据集实现
包含自定义数据集类、数据预处理和批处理逻辑
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from config import config

class TranslationDataset(Dataset):
    """机器翻译数据集类"""
    
    def __init__(self, data_path: str, vocab_zh: Optional[Dict] = None, 
                 vocab_en: Optional[Dict] = None, max_length: int = 100):
        """
        初始化数据集
        
        参数:
            data_path: 数据文件路径
            vocab_zh: 中文词表（如果为None，则从数据构建）
            vocab_en: 英文词表（如果为None，则从数据构建）
            max_length: 最大序列长度
        """
        self.max_length = max_length
        
        # 加载预处理后的数据
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.zh_encoded = data['zh_encoded']
        self.en_encoded = data['en_encoded']
        
        # 词表处理
        if vocab_zh is None or vocab_en is None:
            self.vocab_zh, self.vocab_en = self._build_vocab()
        else:
            self.vocab_zh = vocab_zh
            self.vocab_en = vocab_en
        
        # 添加特殊标记
        self.pad_idx = 0
        self.unk_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3
        
        # 过滤过长的句子
        self._filter_long_sentences()
        
        print(f"数据集加载完成: {len(self)} 个样本")
        print(f"中文词表大小: {len(self.vocab_zh)}")
        print(f"英文词表大小: {len(self.vocab_en)}")
    
    def _build_vocab(self) -> Tuple[Dict, Dict]:
        """从数据构建词表"""
        # 这里使用预处理的词表，实际从数据中统计
        vocab_zh = {}
        vocab_en = {}
        
        # 添加特殊标记
        special_tokens = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        
        # 合并特殊标记
        vocab_zh.update(special_tokens)
        vocab_en.update(special_tokens)
        
        return vocab_zh, vocab_en
    
    def _filter_long_sentences(self):
        """过滤过长的句子"""
        filtered_indices = []
        for i, (zh_seq, en_seq) in enumerate(zip(self.zh_encoded, self.en_encoded)):
            if len(zh_seq) <= self.max_length and len(en_seq) <= self.max_length:
                filtered_indices.append(i)
        
        self.zh_encoded = [self.zh_encoded[i] for i in filtered_indices]
        self.en_encoded = [self.en_encoded[i] for i in filtered_indices]
        
        print(f"过滤后保留 {len(self)}/{len(filtered_indices)} 个样本")
    
    def __len__(self) -> int:
        return len(self.zh_encoded)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        zh_seq = self.zh_encoded[idx]
        en_seq = self.en_encoded[idx]
        
        # 添加起始和结束标记
        zh_tensor = torch.tensor([self.sos_idx] + zh_seq + [self.eos_idx], dtype=torch.long)
        en_tensor = torch.tensor([self.sos_idx] + en_seq + [self.eos_idx], dtype=torch.long)
        
        # 截断到最大长度
        if len(zh_tensor) > self.max_length:
            zh_tensor = zh_tensor[:self.max_length]
            zh_tensor[-1] = self.eos_idx
        
        if len(en_tensor) > self.max_length:
            en_tensor = en_tensor[:self.max_length]
            en_tensor[-1] = self.eos_idx
        
        return {
            'src': zh_tensor,
            'tgt': en_tensor,
            'src_len': len(zh_tensor),
            'tgt_len': len(en_tensor)
        }
    
    def get_vocab_sizes(self) -> Tuple[int, int]:
        """获取词表大小"""
        return len(self.vocab_zh), len(self.vocab_en)
    
    def decode_sequence(self, token_ids: List[int], lang: str = 'en') -> str:
        """将ID序列解码为文本"""
        if lang == 'zh':
            vocab = self.vocab_zh
            inv_vocab = {v: k for k, v in vocab.items()}
        else:
            vocab = self.vocab_en
            inv_vocab = {v: k for k, v in vocab.items()}
        
        # 移除特殊标记并解码
        tokens = []
        for token_id in token_ids:
            if token_id in [self.pad_idx, self.sos_idx, self.eos_idx]:
                continue
            tokens.append(inv_vocab.get(token_id, f'<unk-{token_id}>'))
        
        return ' '.join(tokens)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    批处理函数
    
    参数:
        batch: 批数据列表
        
    返回:
        批处理后的数据字典
    """
    src_sequences = [item['src'] for item in batch]
    tgt_sequences = [item['tgt'] for item in batch]
    src_lengths = torch.tensor([item['src_len'] for item in batch])
    tgt_lengths = torch.tensor([item['tgt_len'] for item in batch])
    
    # 填充序列
    src_padded = pad_sequence(src_sequences, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_sequences, batch_first=True, padding_value=0)
    
    # 对源序列长度进行排序（用于pack_padded_sequence）
    src_lengths, sorted_indices = src_lengths.sort(descending=True)
    src_padded = src_padded[sorted_indices]
    tgt_padded = tgt_padded[sorted_indices]
    tgt_lengths = tgt_lengths[sorted_indices]
    
    return {
        'src': src_padded,
        'tgt': tgt_padded,
        'src_lengths': src_lengths,
        'tgt_lengths': tgt_lengths,
        'sorted_indices': sorted_indices
    }


def create_dataloaders(batch_size: int = 32, max_length: int = 100, 
                      use_bpe: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader, Dict, Dict]:
    """
    创建数据加载器
    
    参数:
        batch_size: 批大小
        max_length: 最大序列长度
        use_bpe: 是否使用BPE分词
        
    返回:
        train_loader: 训练数据加载器
        valid_loader: 验证数据加载器  
        test_loader: 测试数据加载器
        vocab_zh: 中文词表
        vocab_en: 英文词表
    """
    # 确定数据文件路径
    if use_bpe:
        train_path = "data/train_large_processed.pkl"
    else:
        train_path = "data/train_small_processed.pkl"
    
    valid_path = "data/valid_processed.pkl"
    test_path = "data/test_processed.pkl"
    
    # 创建训练集（从数据构建词表）
    train_dataset = TranslationDataset(train_path, max_length=max_length)
    vocab_zh, vocab_en = train_dataset.get_vocab_sizes()
    
    # 创建验证集和测试集（使用训练集的词表）
    valid_dataset = TranslationDataset(valid_path, vocab_zh, vocab_en, max_length=max_length)
    test_dataset = TranslationDataset(test_path, vocab_zh, vocab_en, max_length=max_length)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, valid_loader, test_loader, vocab_zh, vocab_en


class DataManager:
    """数据管理器，提供数据加载和预处理功能"""
    
    def __init__(self, config):
        self.config = config
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.vocab_zh = None
        self.vocab_en = None
    
    def load_data(self):
        """加载所有数据"""
        print("加载数据集...")
        self.train_loader, self.valid_loader, self.test_loader, self.vocab_zh, self.vocab_en = \
            create_dataloaders(
                batch_size=self.config.training.batch_size,
                max_length=self.config.data.max_seq_len,
                use_bpe=(self.config.data.tokenizer_type == "bpe")
            )
        
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"验证集大小: {len(self.valid_loader.dataset)}")
        print(f"测试集大小: {len(self.test_loader.dataset)}")
        
        return self.train_loader, self.valid_loader, self.test_loader
    
    def get_vocab_info(self):
        """获取词表信息"""
        return {
            'src_vocab_size': self.vocab_zh,
            'tgt_vocab_size': self.vocab_en,
            'pad_idx': 0,
            'unk_idx': 1,
            'sos_idx': 2,
            'eos_idx': 3
        }
    
    def show_batch_sample(self, batch_idx: int = 0):
        """显示批次样本示例"""
        if self.train_loader is None:
            print("请先加载数据")
            return
        
        # 获取第一个批次
        for i, batch in enumerate(self.train_loader):
            if i == batch_idx:
                print(f"批次 {batch_idx} 示例:")
                print(f"源序列形状: {batch['src'].shape}")
                print(f"目标序列形状: {batch['tgt'].shape}")
                print(f"源序列长度: {batch['src_lengths']}")
                print(f"目标序列长度: {batch['tgt_lengths']}")
                
                # 显示第一个样本
                src_sample = batch['src'][0].tolist()
                tgt_sample = batch['tgt'][0].tolist()
                
                print(f"\n源序列 (ID): {src_sample[:10]}...")
                print(f"目标序列 (ID): {tgt_sample[:10]}...")
                
                # 解码为文本
                dataset = self.train_loader.dataset
                src_text = dataset.decode_sequence(src_sample, 'zh')
                tgt_text = dataset.decode_sequence(tgt_sample, 'en')
                
                print(f"\n源序列 (文本): {src_text[:50]}...")
                print(f"目标序列 (文本): {tgt_text[:50]}...")
                break


# 测试代码
if __name__ == "__main__":
    # 测试数据集类
    print("测试数据集类...")
    
    # 创建虚拟数据文件用于测试
    test_data = {
        'zh_encoded': [[4, 5, 6, 7, 8], [9, 10, 11], [12, 13]],
        'en_encoded': [[14, 15, 16], [17, 18, 19, 20], [21, 22]]
    }
    
    with open('test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    
    # 创建数据集
    dataset = TranslationDataset('test_data.pkl', max_length=10)
    
    print(f"数据集大小: {len(dataset)}")
    
    # 测试单个样本
    sample = dataset[0]
    print(f"样本键: {list(sample.keys())}")
    print(f"源序列: {sample['src']}")
    print(f"目标序列: {sample['tgt']}")
    print(f"源序列长度: {sample['src_len']}")
    print(f"目标序列长度: {sample['tgt_len']}")
    
    # 测试批处理函数
    batch = [dataset[i] for i in range(3)]
    processed_batch = collate_fn(batch)
    
    print(f"\n批处理后源序列形状: {processed_batch['src'].shape}")
    print(f"批处理后目标序列形状: {processed_batch['tgt'].shape}")
    print(f"源序列长度: {processed_batch['src_lengths']}")
    
    # 测试数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"\n数据加载器批次数量: {len(dataloader)}")
    for i, batch in enumerate(dataloader):
        print(f"批次 {i}:")
        print(f"  源序列形状: {batch['src'].shape}")
        print(f"  目标序列形状: {batch['tgt'].shape}")
    
    # 测试数据管理器
    print("\n测试数据管理器...")
    
    class TestConfig:
        class Training:
            batch_size = 2
        class Data:
            max_seq_len = 100
            tokenizer_type = "word"
    
    config = TestConfig()
    config.training = TestConfig.Training()
    config.data = TestConfig.Data()
    
    data_manager = DataManager(config)
    train_loader, valid_loader, test_loader = data_manager.load_data()
    
    print(f"训练加载器批次: {len(train_loader)}")
    print(f"验证加载器批次: {len(valid_loader)}")
    print(f"测试加载器批次: {len(test_loader)}")
    
    # 清理测试文件
    import os
    if os.path.exists('test_data.pkl'):
        os.remove('test_data.pkl')
    
    print("\n所有测试完成!")