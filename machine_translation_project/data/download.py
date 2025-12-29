import os
import json
import requests
from tqdm import tqdm
from config import config

class DataDownloader:
    """数据下载器"""
    
    def __init__(self):
        self.data_dir = config.data.data_dir
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_from_url(self, url: str, save_path: str):
        """从URL下载文件"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f, tqdm(
            desc=f"Downloading {os.path.basename(save_path)}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                pbar.update(len(data))
    
    def create_dummy_data(self):
        """创建虚拟数据（如果真实数据不可用）"""
        print("创建虚拟数据...")
        
        # 训练数据（小型）
        train_small = []
        for i in range(1000):  # 1000个样本
            zh_text = f"这是第{i}个中文句子。"
            en_text = f"This is the {i}th English sentence."
            train_small.append({"zh": zh_text, "en": en_text})
        
        with open(config.data.train_small_path, 'w', encoding='utf-8') as f:
            for item in train_small:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # 训练数据（大型）
        train_large = []
        for i in range(10000):  # 10000个样本  
            zh_text = f"这是第{i}个更长的中文句子，用于测试机器翻译模型。"
            en_text = f"This is the {i}th longer English sentence for testing machine translation models."
            train_large.append({"zh": zh_text, "en": en_text})
        
        with open(config.data.train_large_path, 'w', encoding='utf-8') as f:
            for item in train_large:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # 验证数据
        valid_data = []
        for i in range(500):
            zh_text = f"验证句子{i}：这是用于验证的中文文本。"
            en_text = f"Validation sentence {i}: This is English text for validation."
            valid_data.append({"zh": zh_text, "en": en_text})
        
        with open(config.data.valid_path, 'w', encoding='utf-8') as f:
            for item in valid_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # 测试数据
        test_data = []
        for i in range(200):
            zh_text = f"测试句子{i}：这是用于测试的中文文本。"
            en_text = f"Test sentence {i}: This is English text for testing."
            test_data.append({"zh": zh_text, "en": en_text})
        
        with open(config.data.test_path, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print("虚拟数据创建完成！")

def download_data():
    """下载数据主函数"""
    downloader = DataDownloader()
    
    # 检查数据是否存在
    if not os.path.exists(config.data.train_small_path):
        print("数据文件不存在，创建虚拟数据...")
        downloader.create_dummy_data()
    else:
        print("数据文件已存在，跳过下载。")

if __name__ == "__main__":
    download_data()