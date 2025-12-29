import os
import json
import re
import jieba
from collections import Counter
from typing import List, Tuple, Dict, Any
import sentencepiece as spm
from nltk.tokenize import word_tokenize
import nltk
from config import config

# 下载NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextCleaner:
    """文本清洗器"""
    
    @staticmethod
    def clean_chinese_text(text: str) -> str:
        """清洗中文文本"""
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        # 保留中文、英文、数字和常用标点
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；："\"\'\"\'-]', ' ', text)
        return text.strip()
    
    @staticmethod
    def clean_english_text(text: str) -> str:
        """清洗英文文本"""
        # 转换为小写
        text = text.lower()
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        # 保留字母、数字和常用标点
        text = re.sub(r'[^a-zA-Z0-9\s\.,!?;:"\'-]', ' ', text)
        return text.strip()
    
    @staticmethod
    def filter_long_sentences(zh_text: str, en_text: str, max_len: int = 100) -> bool:
        """过滤过长的句子"""
        return len(zh_text) <= max_len and len(en_text.split()) <= max_len

class Tokenizer:
    """分词器基类"""
    
    def __init__(self):
        self.vocab = {}
        self.inv_vocab = {}
        
    def build_vocab(self, texts: List[str], max_size: int = 30000, min_freq: int = 2):
        """构建词表"""
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)
        
        # 保留高频词
        vocab_items = [('<pad>', 0), ('<unk>', 1), ('<sos>', 2), ('<eos>', 3)]
        vocab_items.extend([(word, count) for word, count in counter.most_common(max_size) 
                           if count >= min_freq])
        
        self.vocab = {word: idx for idx, (word, _) in enumerate(vocab_items)}
        self.inv_vocab = {idx: word for word, idx in self.vocab.items()}
        
        return self.vocab
    
    def tokenize(self, text: str) -> List[str]:
        """分词（需子类实现）"""
        raise NotImplementedError
    
    def encode(self, text: str) -> List[int]:
        """编码文本为ID序列"""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab['<unk>']) for token in tokens] + [self.vocab['<eos>']]
    
    def decode(self, token_ids: List[int]) -> str:
        """解码ID序列为文本"""
        tokens = [self.inv_vocab.get(idx, '<unk>') for idx in token_ids]
        # 移除特殊标记
        tokens = [t for t in tokens if t not in ['<sos>', '<eos>', '<pad>']]
        return ' '.join(tokens)

class ChineseTokenizer(Tokenizer):
    """中文分词器"""
    
    def __init__(self, use_jieba: bool = True):
        super().__init__()
        self.use_jieba = use_jieba
        
    def tokenize(self, text: str) -> List[str]:
        """中文分词"""
        if self.use_jieba:
            return list(jieba.cut(text))
        else:
            # 字符级分词
            return list(text)

class EnglishTokenizer(Tokenizer):
    """英文分词器"""
    
    def tokenize(self, text: str) -> List[str]:
        """英文分词"""
        return word_tokenize(text)

class BPETokenizer:
    """BPE分词器"""
    
    def __init__(self):
        self.sp_model = None
        
    def train(self, texts: List[str], vocab_size: int = 30000, model_prefix: str = "bpe_model"):
        """训练BPE模型"""
        # 准备训练数据
        temp_file = "temp_training_data.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        # 训练SentencePiece模型
        spm.SentencePieceTrainer.train(
            f'--input={temp_file} '
            f'--model_prefix={model_prefix} '
            f'--vocab_size={vocab_size} '
            f'--character_coverage=1.0 '
            f'--model_type=bpe '
            f'--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3'
        )
        
        # 加载模型
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(f'{model_prefix}.model')
        
        # 清理临时文件
        os.remove(temp_file)
    
    def encode(self, text: str) -> List[int]:
        """编码文本"""
        return self.sp_model.encode_as_ids(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """解码ID序列"""
        return self.sp_model.decode_ids(token_ids)

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        self.cleaner = TextCleaner()
        self.zh_tokenizer = ChineseTokenizer()
        self.en_tokenizer = EnglishTokenizer()
        self.bpe_tokenizer = None
        
    def load_data(self, file_path: str) -> List[Tuple[str, str]]:
        """加载数据"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append((item['zh'], item['en']))
        return data
    
    def preprocess_data(self, data: List[Tuple[str, str]], use_bpe: bool = False) -> Tuple[List[List[int]], List[List[int]], Dict, Dict]:
        """预处理数据"""
        zh_texts, en_texts = zip(*data)
        
        # 清洗文本
        zh_clean = [self.cleaner.clean_chinese_text(text) for text in zh_texts]
        en_clean = [self.cleaner.clean_english_text(text) for text in en_texts]
        
        # 过滤长句子
        filtered_data = []
        for zh, en in zip(zh_clean, en_clean):
            if self.cleaner.filter_long_sentences(zh, en, config.data.max_seq_len):
                filtered_data.append((zh, en))
        
        print(f"过滤后保留 {len(filtered_data)}/{len(data)} 个样本")
        
        zh_filtered, en_filtered = zip(*filtered_data)
        
        if use_bpe:
            # 使用BPE分词
            if self.bpe_tokenizer is None:
                self.bpe_tokenizer = BPETokenizer()
                # 合并所有文本训练BPE模型
                all_texts = list(zh_filtered) + list(en_filtered)
                self.bpe_tokenizer.train(all_texts, config.data.vocab_size)
            
            zh_encoded = [self.bpe_tokenizer.encode(text) for text in zh_filtered]
            en_encoded = [self.bpe_tokenizer.encode(text) for text in en_filtered]
            
            # 创建虚拟词表（BPE不需要传统词表）
            vocab_zh = {"bpe": self.bpe_tokenizer}
            vocab_en = {"bpe": self.bpe_tokenizer}
            
        else:
            # 使用传统分词
            # 构建词表
            vocab_zh = self.zh_tokenizer.build_vocab(zh_filtered, config.data.vocab_size)
            vocab_en = self.en_tokenizer.build_vocab(en_filtered, config.data.vocab_size)
            
            # 编码文本
            zh_encoded = [self.zh_tokenizer.encode(text) for text in zh_filtered]
            en_encoded = [self.en_tokenizer.encode(text) for text in en_filtered]
        
        return zh_encoded, en_encoded, vocab_zh, vocab_en
    
    def save_processed_data(self, data: Tuple, save_path: str):
        """保存处理后的数据"""
        zh_encoded, en_encoded, vocab_zh, vocab_en = data
        
        processed_data = {
            'zh_encoded': zh_encoded,
            'en_encoded': en_encoded,
            'vocab_zh': vocab_zh,
            'vocab_en': vocab_en
        }
        
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(processed_data, f)
    
    def load_processed_data(self, load_path: str) -> Tuple:
        """加载处理后的数据"""
        import pickle
        with open(load_path, 'rb') as f:
            return pickle.load(f)

def preprocess_all_data():
    """预处理所有数据"""
    preprocessor = DataPreprocessor()
    
    # 处理小型训练集
    print("处理小型训练集...")
    train_small_data = preprocessor.load_data(config.data.train_small_path)
    train_small_processed = preprocessor.preprocess_data(train_small_data, use_bpe=False)
    preprocessor.save_processed_data(train_small_processed, "data/train_small_processed.pkl")
    
    # 处理大型训练集
    print("处理大型训练集...")
    train_large_data = preprocessor.load_data(config.data.train_large_path)
    train_large_processed = preprocessor.preprocess_data(train_large_data, use_bpe=True)
    preprocessor.save_processed_data(train_large_processed, "data/train_large_processed.pkl")
    
    # 处理验证集
    print("处理验证集...")
    valid_data = preprocessor.load_data(config.data.valid_path)
    valid_processed = preprocessor.preprocess_data(valid_data, use_bpe=False)
    preprocessor.save_processed_data(valid_processed, "data/valid_processed.pkl")
    
    # 处理测试集
    print("处理测试集...")
    test_data = preprocessor.load_data(config.data.test_path)
    test_processed = preprocessor.preprocess_data(test_data, use_bpe=False)
    preprocessor.save_processed_data(test_processed, "data/test_processed.pkl")
    
    print("数据预处理完成！")

if __name__ == "__main__":
    preprocess_all_data()