"""
机器翻译评估指标计算
包含BLEU、ROUGE、METEOR等常见翻译质量评估指标
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import math
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import jieba
import sacrebleu

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')

class TranslationMetrics:
    """翻译质量评估指标"""
    
    def __init__(self):
        self.rouge = Rouge()
        self.smoothing_function = SmoothingFunction().method1
    
    def calculate_bleu(self, references: List[List[str]], candidates: List[str], 
                      weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)) -> Dict[str, float]:
        """
        计算BLEU分数
        
        参数:
            references: 参考翻译列表，每个元素是一个参考翻译的token列表
            candidates: 候选翻译列表，每个元素是一个候选翻译的token列表
            weights: BLEU的n-gram权重
            
        返回:
            bleu_scores: 包含各种BLEU分数的字典
        """
        # 句子级BLEU
        sentence_bleus = []
        for refs, cand in zip(references, candidates):
            if not cand:  # 空候选翻译
                sentence_bleus.append(0.0)
                continue
            
            # 将字符串转换为token列表
            if isinstance(cand, str):
                cand_tokens = cand.split()
            else:
                cand_tokens = cand
            
            refs_tokens = []
            for ref in refs:
                if isinstance(ref, str):
                    refs_tokens.append(ref.split())
                else:
                    refs_tokens.append(ref)
            
            try:
                score = sentence_bleu(refs_tokens, cand_tokens, 
                                     weights=weights,
                                     smoothing_function=self.smoothing_function)
                sentence_bleus.append(score)
            except:
                sentence_bleus.append(0.0)
        
        # 语料级BLEU（使用sacrebleu）
        if all(isinstance(c, str) for c in candidates) and all(isinstance(r, str) for refs in references for r in refs):
            cand_corpus = candidates
            ref_corpus = [' '.join(refs) for refs in references]
            
            # 计算语料级BLEU
            bleu = sacrebleu.corpus_bleu(cand_corpus, [ref_corpus])
            
            bleu_scores = {
                'bleu': bleu.score,
                'bleu_1': np.mean([sentence_bleu(refs, cand, weights=(1, 0, 0, 0), 
                                                smoothing_function=self.smoothing_function) 
                                  for refs, cand in zip(references, candidates)]),
                'bleu_2': np.mean([sentence_bleu(refs, cand, weights=(0.5, 0.5, 0, 0), 
                                                smoothing_function=self.smoothing_function) 
                                  for refs, cand in zip(references, candidates)]),
                'bleu_3': np.mean([sentence_bleu(refs, cand, weights=(0.33, 0.33, 0.33, 0), 
                                                smoothing_function=self.smoothing_function) 
                                  for refs, cand in zip(references, candidates)]),
                'bleu_4': np.mean([sentence_bleu(refs, cand, weights=weights, 
                                                smoothing_function=self.smoothing_function) 
                                  for refs, cand in zip(references, candidates)]),
                'sentence_bleu_mean': np.mean(sentence_bleus),
                'sentence_bleu_std': np.std(sentence_bleus)
            }
        else:
            bleu_scores = {
                'bleu_1': np.mean([sentence_bleu(refs, cand, weights=(1, 0, 0, 0), 
                                                smoothing_function=self.smoothing_function) 
                                  for refs, cand in zip(references, candidates)]),
                'bleu_2': np.mean([sentence_bleu(refs, cand, weights=(0.5, 0.5, 0, 0), 
                                                smoothing_function=self.smoothing_function) 
                                  for refs, cand in zip(references, candidates)]),
                'bleu_3': np.mean([sentence_bleu(refs, cand, weights=(0.33, 0.33, 0.33, 0), 
                                                smoothing_function=self.smoothing_function) 
                                  for refs, cand in zip(references, candidates)]),
                'bleu_4': np.mean([sentence_bleu(refs, cand, weights=weights, 
                                                smoothing_function=self.smoothing_function) 
                                  for refs, cand in zip(references, candidates)]),
                'sentence_bleu_mean': np.mean(sentence_bleus),
                'sentence_bleu_std': np.std(sentence_bleus)
            }
        
        return bleu_scores
    
    def calculate_rouge(self, references: List[List[str]], candidates: List[str]) -> Dict[str, float]:
        """
        计算ROUGE分数
        
        参数:
            references: 参考翻译列表
            candidates: 候选翻译列表
            
        返回:
            rouge_scores: ROUGE分数字典
        """
        # 准备ROUGE输入格式
        refs_for_rouge = []
        cands_for_rouge = []
        
        for refs, cand in zip(references, candidates):
            if isinstance(cand, str):
                cand_text = cand
            else:
                cand_text = ' '.join(cand)
            
            # 使用第一个参考翻译（ROUGE通常只接受一个参考）
            if isinstance(refs[0], str):
                ref_text = refs[0]
            else:
                ref_text = ' '.join(refs[0])
            
            refs_for_rouge.append(ref_text)
            cands_for_rouge.append(cand_text)
        
        try:
            scores = self.rouge.get_scores(cands_for_rouge, refs_for_rouge, avg=True)
            rouge_scores = {
                'rouge-1': {
                    'f': scores['rouge-1']['f'],
                    'p': scores['rouge-1']['p'],
                    'r': scores['rouge-1']['r']
                },
                'rouge-2': {
                    'f': scores['rouge-2']['f'],
                    'p': scores['rouge-2']['p'],
                    'r': scores['rouge-2']['r']
                },
                'rouge-l': {
                    'f': scores['rouge-l']['f'],
                    'p': scores['rouge-l']['p'],
                    'r': scores['rouge-l']['r']
                }
            }
        except:
            # 如果计算失败，返回默认值
            rouge_scores = {
                'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
            }
        
        return rouge_scores
    
    def calculate_meteor(self, references: List[List[str]], candidates: List[str]) -> Dict[str, float]:
        """
        计算METEOR分数
        
        参数:
            references: 参考翻译列表
            candidates: 候选翻译列表
            
        返回:
            meteor_scores: METEOR分数字典
        """
        meteor_scores = []
        
        for refs, cand in zip(references, candidates):
            if isinstance(cand, str):
                cand_tokens = cand.split()
            else:
                cand_tokens = cand
            
            # 选择第一个参考翻译
            if isinstance(refs[0], str):
                ref_tokens = refs[0].split()
            else:
                ref_tokens = refs[0]
            
            try:
                score = meteor_score([ref_tokens], cand_tokens)
                meteor_scores.append(score)
            except:
                meteor_scores.append(0.0)
        
        return {
            'meteor_mean': np.mean(meteor_scores),
            'meteor_std': np.std(meteor_scores),
            'meteor_scores': meteor_scores
        }
    
    def calculate_chrF(self, references: List[List[str]], candidates: List[str]) -> Dict[str, float]:
        """
        计算chrF分数（字符级F-score）
        
        参数:
            references: 参考翻译列表
            candidates: 候选翻译列表
            
        返回:
            chrf_scores: chrF分数字典
        """
        try:
            # 准备sacrebleu格式
            cand_corpus = [cand if isinstance(cand, str) else ' '.join(cand) for cand in candidates]
            ref_corpus = []
            for refs in references:
                if isinstance(refs[0], str):
                    ref_corpus.append(refs[0])
                else:
                    ref_corpus.append(' '.join(refs[0]))
            
            # 计算chrF++
            chrf = sacrebleu.corpus_chrf(cand_corpus, [ref_corpus])
            
            return {
                'chrf': chrf.score,
                'chrf_score': chrf.score  # 兼容性
            }
        except:
            return {'chrf': 0.0, 'chrf_score': 0.0}
    
    def calculate_ter(self, references: List[List[str]], candidates: List[str]) -> Dict[str, float]:
        """
        计算TER（翻译编辑率）
        
        参数:
            references: 参考翻译列表
            candidates: 候选翻译列表
            
        返回:
            ter_scores: TER分数字典
        """
        try:
            # 准备sacrebleu格式
            cand_corpus = [cand if isinstance(cand, str) else ' '.join(cand) for cand in candidates]
            ref_corpus = []
            for refs in references:
                if isinstance(refs[0], str):
                    ref_corpus.append(refs[0])
                else:
                    ref_corpus.append(' '.join(refs[0]))
            
            # 计算TER
            ter = sacrebleu.corpus_ter(cand_corpus, [ref_corpus])
            
            return {
                'ter': ter.score,
                'ter_score': ter.score  # 兼容性
            }
        except:
            return {'ter': 100.0, 'ter_score': 100.0}
    
    def calculate_all_metrics(self, references: List[List[str]], candidates: List[str]) -> Dict[str, Any]:
        """
        计算所有评估指标
        
        参数:
            references: 参考翻译列表
            candidates: 候选翻译列表
            
        返回:
            all_metrics: 包含所有指标的字典
        """
        print("计算BLEU分数...")
        bleu_scores = self.calculate_bleu(references, candidates)
        
        print("计算ROUGE分数...")
        rouge_scores = self.calculate_rouge(references, candidates)
        
        print("计算METEOR分数...")
        meteor_scores = self.calculate_meteor(references, candidates)
        
        print("计算chrF分数...")
        chrf_scores = self.calculate_chrF(references, candidates)
        
        print("计算TER分数...")
        ter_scores = self.calculate_ter(references, candidates)
        
        # 合并所有指标
        all_metrics = {
            'bleu': bleu_scores,
            'rouge': rouge_scores,
            'meteor': meteor_scores,
            'chrf': chrf_scores,
            'ter': ter_scores
        }
        
        return all_metrics
    
    def evaluate_model_outputs(self, model_outputs: List[str], 
                              reference_texts: List[str],
                              source_lang: str = 'zh',
                              target_lang: str = 'en') -> Dict[str, Any]:
        """
        评估模型输出
        
        参数:
            model_outputs: 模型生成的翻译
            reference_texts: 参考翻译
            source_lang: 源语言
            target_lang: 目标语言
            
        返回:
            evaluation_results: 评估结果
        """
        # 准备参考翻译格式
        references = [[ref] for ref in reference_texts]
        
        # 分词处理
        if target_lang == 'zh':
            # 中文分词
            candidates = [' '.join(jieba.cut(output)) for output in model_outputs]
            references = [[' '.join(jieba.cut(ref[0]))] for ref in references]
        else:
            # 英文分词
            candidates = [output.split() for output in model_outputs]
            references = [[ref[0].split()] for ref in references]
        
        # 计算所有指标
        metrics = self.calculate_all_metrics(references, candidates)
        
        # 添加基本信息
        evaluation_results = {
            'num_samples': len(model_outputs),
            'source_lang': source_lang,
            'target_lang': target_lang,
            'metrics': metrics,
            'model_outputs': model_outputs,
            'references': reference_texts
        }
        
        return evaluation_results
    
    def print_metrics_summary(self, metrics: Dict[str, Any]):
        """打印指标摘要"""
        print("\n" + "="*60)
        print("翻译质量评估结果")
        print("="*60)
        
        # BLEU分数
        if 'bleu' in metrics:
            bleu = metrics['bleu']
            print(f"\nBLEU分数:")
            print(f"  BLEU: {bleu.get('bleu', bleu.get('bleu_4', 0)):.2f}")
            print(f"  BLEU-1: {bleu.get('bleu_1', 0):.2f}")
            print(f"  BLEU-2: {bleu.get('bleu_2', 0):.2f}")
            print(f"  BLEU-3: {bleu.get('bleu_3', 0):.2f}")
            print(f"  BLEU-4: {bleu.get('bleu_4', 0):.2f}")
            print(f"  句子BLEU均值: {bleu.get('sentence_bleu_mean', 0):.2f} ± {bleu.get('sentence_bleu_std', 0):.2f}")
        
        # ROUGE分数
        if 'rouge' in metrics:
            rouge = metrics['rouge']
            print(f"\nROUGE分数:")
            for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
                if metric in rouge:
                    print(f"  {metric.upper()}: F={rouge[metric]['f']:.3f}, P={rouge[metric]['p']:.3f}, R={rouge[metric]['r']:.3f}")
        
        # METEOR分数
        if 'meteor' in metrics:
            meteor = metrics['meteor']
            print(f"\nMETEOR分数:")
            print(f"  METEOR: {meteor.get('meteor_mean', 0):.3f} ± {meteor.get('meteor_std', 0):.3f}")
        
        # chrF分数
        if 'chrf' in metrics:
            chrf = metrics['chrf']
            print(f"\nchrF分数:")
            print(f"  chrF: {chrf.get('chrf', 0):.2f}")
        
        # TER分数
        if 'ter' in metrics:
            ter = metrics['ter']
            print(f"\nTER分数:")
            print(f"  TER: {ter.get('ter', 0):.2f} (越低越好)")
        
        print("="*60)


# 测试代码
if __name__ == "__main__":
    # 创建评估器
    evaluator = TranslationMetrics()
    
    # 测试数据
    references = [
        ["the cat is on the mat"],
        ["hello world"],
        ["machine translation is difficult"]
    ]
    
    candidates = [
        "cat is on mat",
        "hello world",
        "machine translation is hard"
    ]
    
    # 计算所有指标
    print("计算翻译评估指标...")
    metrics = evaluator.calculate_all_metrics(references, candidates)
    
    # 打印结果
    evaluator.print_metrics_summary(metrics)
    
    # 测试模型输出评估
    print("\n测试模型输出评估...")
    model_outputs = [
        "cat is on mat",
        "hello world", 
        "machine translation is hard"
    ]
    reference_texts = [
        "the cat is on the mat",
        "hello world",
        "machine translation is difficult"
    ]
    
    evaluation_results = evaluator.evaluate_model_outputs(
        model_outputs, reference_texts, source_lang='zh', target_lang='en'
    )
    
    print(f"\n评估完成，共处理 {evaluation_results['num_samples']} 个样本")
    
    print("\n所有测试完成!")