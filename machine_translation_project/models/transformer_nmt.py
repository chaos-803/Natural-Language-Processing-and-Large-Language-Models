"""
基于Transformer的神经机器翻译模型实现
包含多头注意力、位置编码、编码器、解码器和完整Transformer架构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Optional, Tuple, Dict, Any
import numpy as np

class PositionalEncoding(nn.Module):
    """正弦余弦位置编码"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )  # [d_model/2]
        
        # 计算正弦和余弦位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦
        
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        
        参数:
            x: 输入序列 [seq_len, batch_size, d_model]
            
        返回:
            添加位置编码后的序列 [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]  # 只取前seq_len个位置
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0, "d_model必须能被nhead整除"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)  # 查询变换
        self.w_k = nn.Linear(d_model, d_model)  # 键变换
        self.w_v = nn.Linear(d_model, d_model)  # 值变换
        self.w_o = nn.Linear(d_model, d_model)   # 输出变换
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        多头注意力前向传播
        
        参数:
            query: 查询张量 [seq_len_q, batch_size, d_model] 或 [batch_size, seq_len_q, d_model]
            key: 键张量 [seq_len_k, batch_size, d_model] 或 [batch_size, seq_len_k, d_model]
            value: 值张量 [seq_len_v, batch_size, d_model] 或 [batch_size, seq_len_v, d_model]
            key_padding_mask: 键填充掩码 [batch_size, seq_len_k]
            attn_mask: 注意力掩码 [seq_len_q, seq_len_k] 或 [batch_size, seq_len_q, seq_len_k]
            
        返回:
            output: 注意力输出 [seq_len_q, batch_size, d_model]
            attn_weights: 注意力权重 [batch_size, nhead, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0) if query.dim() == 3 else query.size(1)
        seq_len_q = query.size(1) if query.dim() == 3 else query.size(0)
        seq_len_k = key.size(1) if key.dim() == 3 else key.size(0)
        
        # 线性变换并分头
        Q = self.w_q(query)  # [batch_size, seq_len_q, d_model] 或 [seq_len_q, batch_size, d_model]
        K = self.w_k(key)    # [batch_size, seq_len_k, d_model] 或 [seq_len_k, batch_size, d_model]
        V = self.w_v(value)  # [batch_size, seq_len_v, d_model] 或 [seq_len_v, batch_size, d_model]
        
        # 调整维度以适应多头注意力
        if Q.dim() == 3:  # [batch_size, seq_len, d_model]
            Q = Q.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)  # [batch_size, nhead, seq_len_q, d_k]
            K = K.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)  # [batch_size, nhead, seq_len_k, d_k]
            V = V.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)  # [batch_size, nhead, seq_len_v, d_k]
        else:  # [seq_len, batch_size, d_model]
            Q = Q.view(-1, batch_size, self.nhead, self.d_k).transpose(0, 1)  # [batch_size, nhead, seq_len_q, d_k]
            K = K.view(-1, batch_size, self.nhead, self.d_k).transpose(0, 1)  # [batch_size, nhead, seq_len_k, d_k]
            V = V.view(-1, batch_size, self.nhead, self.d_k).transpose(0, 1)  # [batch_size, nhead, seq_len_v, d_k]
        
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch_size, nhead, seq_len_q, seq_len_k]
        
        # 应用注意力掩码
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len_q, seq_len_k]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_k]
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)
        
        # 应用键填充掩码
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len_k]
            attn_scores = attn_scores.masked_fill(key_padding_mask, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到值
        output = torch.matmul(attn_weights, V)  # [batch_size, nhead, seq_len_q, d_k]
        
        # 合并多头
        if output.dim() == 4:  # 来自batch-first输入
            output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        else:
            output = output.transpose(0, 1).contiguous().view(-1, batch_size, self.d_model)
        
        # 最终线性变换
        output = self.w_o(output)
        
        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """位置前馈网络"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前馈网络前向传播
        
        参数:
            x: 输入张量 [seq_len, batch_size, d_model] 或 [batch_size, seq_len, d_model]
            
        返回:
            输出张量，形状与输入相同
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # 自注意力子层
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 前馈网络子层
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码器层前向传播
        
        参数:
            src: 输入序列 [seq_len, batch_size, d_model] 或 [batch_size, seq_len, d_model]
            src_mask: 源序列掩码
            src_key_padding_mask: 源键填充掩码
            
        返回:
            编码后的序列
        """
        # 自注意力子层（带残差连接和层归一化）
        attn_output, _ = self.self_attn(src, src, src, 
                                       key_padding_mask=src_key_padding_mask,
                                       attn_mask=src_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        
        # 前馈网络子层（带残差连接和层归一化）
        ffn_output = self.ffn(src)
        src = src + self.dropout2(ffn_output)
        src = self.norm2(src)
        
        return src


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""
    
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super(TransformerDecoderLayer, self).__init__()
        
        # 掩码自注意力子层
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 编码器-解码器注意力子层
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 前馈网络子层
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        解码器层前向传播
        
        参数:
            tgt: 目标序列 [seq_len_t, batch_size, d_model] 或 [batch_size, seq_len_t, d_model]
            memory: 编码器输出 [seq_len_s, batch_size, d_model] 或 [batch_size, seq_len_s, d_model]
            tgt_mask: 目标序列掩码
            memory_mask: 记忆序列掩码
            tgt_key_padding_mask: 目标键填充掩码
            memory_key_padding_mask: 记忆键填充掩码
            
        返回:
            解码后的序列
        """
        # 掩码自注意力子层
        attn_output1, _ = self.self_attn(tgt, tgt, tgt,
                                        key_padding_mask=tgt_key_padding_mask,
                                        attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(attn_output1)
        tgt = self.norm1(tgt)
        
        # 编码器-解码器注意力子层
        attn_output2, _ = self.cross_attn(tgt, memory, memory,
                                         key_padding_mask=memory_key_padding_mask,
                                         attn_mask=memory_mask)
        tgt = tgt + self.dropout2(attn_output2)
        tgt = self.norm2(tgt)
        
        # 前馈网络子层
        ffn_output = self.ffn(tgt)
        tgt = tgt + self.dropout3(ffn_output)
        tgt = self.norm3(tgt)
        
        return tgt


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    
    def __init__(self, encoder_layer: nn.Module, num_layers: int = 6, norm: Optional[nn.Module] = None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码器前向传播
        
        参数:
            src: 输入序列
            mask: 注意力掩码
            src_key_padding_mask: 键填充掩码
            
        返回:
            编码后的序列
        """
        output = src
        
        for layer in self.layers:
            output = layer(output, src_mask=mask, 
                          src_key_padding_mask=src_key_padding_mask)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output


class TransformerDecoder(nn.Module):
    """Transformer解码器"""
    
    def __init__(self, decoder_layer: nn.Module, num_layers: int = 6, norm: Optional[nn.Module] = None):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        解码器前向传播
        
        参数:
            tgt: 目标序列
            memory: 编码器输出
            tgt_mask: 目标掩码
            memory_mask: 记忆掩码
            tgt_key_padding_mask: 目标键填充掩码
            memory_key_padding_mask: 记忆键填充掩码
            
        返回:
            解码后的序列
        """
        output = tgt
        
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output


class TransformerNMT(nn.Module):
    """基于Transformer的神经机器翻译模型"""
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", max_seq_length: int = 5000):
        super(TransformerNMT, self).__init__()
        
        # 模型参数
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # 缩放因子
        self.scale = math.sqrt(d_model)
        
        # 编码器
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        # 解码器
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        
        # 输出层
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 特殊标记索引
        self.pad_idx = 0
        self.sos_idx = 2
        self.eos_idx = 3
        
        # 初始化参数
        self._reset_parameters()
    
    def _reset_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        训练模式的前向传播
        
        参数:
            src: 源语言序列 [batch_size, src_len]
            tgt: 目标语言序列 [batch_size, tgt_len]
            src_mask: 源序列掩码 [src_len, src_len]
            tgt_mask: 目标序列掩码 [tgt_len, tgt_len]
            src_key_padding_mask: 源键填充掩码 [batch_size, src_len]
            tgt_key_padding_mask: 目标键填充掩码 [batch_size, tgt_len]
            memory_key_padding_mask: 记忆键填充掩码 [batch_size, src_len]
            
        返回:
            解码器输出 [batch_size, tgt_len, tgt_vocab_size]
        """
        batch_size, src_len = src.size()
        tgt_len = tgt.size(1)
        
        # 调整维度顺序为 [seq_len, batch_size, d_model]
        src = src.transpose(0, 1)  # [src_len, batch_size]
        tgt = tgt.transpose(0, 1)  # [tgt_len, batch_size]
        
        # 创建目标掩码（防止看到未来信息）
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        
        # 源序列嵌入和位置编码
        src_embed = self.src_embedding(src) * self.scale
        src_embed = self.pos_encoder(src_embed)
        
        # 编码器
        memory = self.encoder(src_embed, mask=src_mask, 
                             src_key_padding_mask=src_key_padding_mask)
        
        # 目标序列嵌入和位置编码
        tgt_embed = self.tgt_embedding(tgt) * self.scale
        tgt_embed = self.pos_decoder(tgt_embed)
        
        # 解码器
        output = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask,
                            memory_mask=None,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)
        
        # 输出投影
        output = self.output_projection(output)  # [tgt_len, batch_size, tgt_vocab_size]
        
        # 调整回 [batch_size, tgt_len, tgt_vocab_size]
        output = output.transpose(0, 1)
        
        return output
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
               src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """编码源序列"""
        src = src.transpose(0, 1)  # [src_len, batch_size]
        src_embed = self.src_embedding(src) * self.scale
        src_embed = self.pos_encoder(src_embed)
        memory = self.encoder(src_embed, mask=src_mask, 
                            src_key_padding_mask=src_key_padding_mask)
        return memory
    
    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """解码目标序列"""
        tgt = tgt.transpose(0, 1)  # [tgt_len, batch_size]
        tgt_embed = self.tgt_embedding(tgt) * self.scale
        tgt_embed = self.pos_decoder(tgt_embed)
        output = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask,
                            memory_key_padding_mask=memory_key_padding_mask)
        output = self.output_projection(output)
        output = output.transpose(0, 1)  # [batch_size, tgt_len, tgt_vocab_size]
        return output
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """生成方形后续掩码，防止解码时看到未来信息"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def create_padding_mask(self, src: torch.Tensor, pad_idx: int) -> torch.Tensor:
        """创建填充掩码"""
        return (src == pad_idx)
    
    def translate(self, src: torch.Tensor, max_len: int = 100) -> torch.Tensor:
        """
        贪婪解码翻译
        
        参数:
            src: 源语言序列 [batch_size, src_len]
            max_len: 最大生成长度
            
        返回:
            translated_ids: 翻译结果 [batch_size, seq_len]
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        # 编码源序列
        src_mask = None
        src_key_padding_mask = self.create_padding_mask(src, self.pad_idx)
        memory = self.encode(src, src_mask, src_key_padding_mask)
        
        # 初始化解码器输入（<SOS>标记）
        tgt = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=device)
        
        # 存储翻译结果
        translated_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        translated_ids[:, 0] = self.sos_idx
        
        # 贪婪解码循环
        for i in range(1, max_len):
            # 创建目标序列掩码
            tgt_mask = self.generate_square_subsequent_mask(i).to(device)
            
            # 解码
            output = self.decode(tgt, memory, tgt_mask, src_key_padding_mask)
            
            # 获取最后一个时间步的输出
            next_token_logits = output[:, -1, :]  # [batch_size, vocab_size]
            
            # 选择概率最高的词
            _, next_token = torch.max(next_token_logits, dim=1)  # [batch_size]
            
            # 添加到目标序列
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
            
            # 存储结果
            translated_ids[:, i] = next_token
            
            # 如果所有序列都生成了<EOS>，则停止
            if (next_token == self.eos_idx).all():
                break
        
        return translated_ids
    
    def beam_search(self, src: torch.Tensor, beam_size: int = 5, 
                    max_len: int = 100, length_penalty: float = 0.6) -> torch.Tensor:
        """
        集束搜索解码
        
        参数:
            src: 源语言序列 [batch_size, src_len]
            beam_size: 集束大小
            max_len: 最大生成长度
            length_penalty: 长度惩罚因子
            
        返回:
            best_sequence: 最佳翻译序列 [batch_size, seq_len]
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        # 编码源序列
        src_key_padding_mask = self.create_padding_mask(src, self.pad_idx)
        memory = self.encode(src, None, src_key_padding_mask)
        
        # 存储每个批次的最终结果
        all_best_sequences = []
        
        # 对每个样本单独进行集束搜索
        for b in range(batch_size):
            # 提取单个样本的编码器输出
            memory_b = memory[:, b:b+1].expand(-1, beam_size, -1)  # [src_len, beam_size, d_model]
            src_key_padding_mask_b = src_key_padding_mask[b:b+1].expand(beam_size, -1)  # [beam_size, src_len]
            
            # 初始化集束
            beams = [{
                'sequence': [self.sos_idx],
                'score': 0.0,
                'tgt': torch.tensor([[self.sos_idx]], device=device)
            }]
            
            completed_beams = []
            
            for step in range(max_len):
                candidates = []
                
                for beam in beams:
                    # 如果已经生成EOS，则不再扩展
                    if beam['sequence'][-1] == self.eos_idx:
                        candidates.append(beam)
                        continue
                    
                    # 获取当前目标序列
                    tgt_len = beam['tgt'].size(1)
                    
                    # 创建目标掩码
                    tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(device)
                    
                    # 解码
                    output = self.decode(beam['tgt'].transpose(0, 1), 
                                        memory_b[:, :1, :],  # 只取第一个集束的memory
                                        tgt_mask, 
                                        src_key_padding_mask_b[:1, :])
                    
                    # 获取最后一个时间步的输出
                    next_token_logits = output[:, -1, :]  # [1, vocab_size]
                    log_probs = F.log_softmax(next_token_logits, dim=-1).squeeze(0)  # [vocab_size]
                    
                    # 获取top-k个候选
                    topk_probs, topk_indices = log_probs.topk(beam_size)
                    
                    for j in range(beam_size):
                        token_id = topk_indices[j].item()
                        token_score = topk_probs[j].item()
                        
                        # 计算新分数（带长度惩罚）
                        new_sequence = beam['sequence'] + [token_id]
                        length = len(new_sequence)
                        score = (beam['score'] * (length - 1) ** length_penalty + token_score) / length ** length_penalty
                        
                        candidate = {
                            'sequence': new_sequence,
                            'score': score,
                            'tgt': torch.cat([beam['tgt'], 
                                            torch.tensor([[token_id]], device=device)], dim=1)
                        }
                        candidates.append(candidate)
                
                if not candidates:
                    break
                
                # 按分数排序并选择top beam_size个
                candidates.sort(key=lambda x: x['score'], reverse=True)
                beams = candidates[:beam_size]
                
                # 检查是否有完成的序列
                new_completed = []
                remaining_beams = []
                
                for beam in beams:
                    if beam['sequence'][-1] == self.eos_idx or len(beam['sequence']) >= max_len:
                        new_completed.append(beam)
                    else:
                        remaining_beams.append(beam)
                
                completed_beams.extend(new_completed)
                beams = remaining_beams
                
                # 如果所有序列都完成，则停止
                if not beams:
                    break
            
            # 选择最佳序列
            all_beams = beams + completed_beams
            if not all_beams:
                # 如果没有生成任何序列，返回空序列
                best_sequence = [self.sos_idx, self.eos_idx]
            else:
                all_beams.sort(key=lambda x: x['score'], reverse=True)
                best_sequence = all_beams[0]['sequence']
            
            # 填充或截断序列
            if len(best_sequence) < max_len:
                best_sequence = best_sequence + [self.pad_idx] * (max_len - len(best_sequence))
            else:
                best_sequence = best_sequence[:max_len]
            
            all_best_sequences.append(best_sequence)
        
        # 转换为张量
        return torch.tensor(all_best_sequences, device=device)


# 测试代码
if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    
    # 测试参数
    batch_size = 4
    src_seq_len = 10
    tgt_seq_len = 12
    src_vocab_size = 5000
    tgt_vocab_size = 6000
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    
    # 创建模型实例
    model = TransformerNMT(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=0.1
    )
    
    # 创建随机输入
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    # 创建掩码
    src_key_padding_mask = (src == 0)
    tgt_key_padding_mask = (tgt == 0)
    
    # 测试前向传播
    print("测试前向传播...")
    outputs = model(src, tgt[:, :-1],  # 训练时使用shifted-right目标序列
                   src_key_padding_mask=src_key_padding_mask,
                   tgt_key_padding_mask=tgt_key_padding_mask[:, :-1])
    print(f"输出形状: {outputs.shape}")  # 应为 [batch_size, tgt_len-1, tgt_vocab_size]
    print(f"输出值范围: [{outputs.min():.4f}, {outputs.max():.4f}]")
    
    # 测试编码器
    print("\n测试编码器...")
    memory = model.encode(src, src_key_padding_mask=src_key_padding_mask)
    print(f"编码器输出形状: {memory.shape}")  # 应为 [src_len, batch_size, d_model]
    
    # 测试解码器
    print("\n测试解码器...")
    tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len - 1)
    decoder_output = model.decode(tgt[:, :-1].transpose(0, 1), memory, 
                                 tgt_mask, src_key_padding_mask)
    print(f"解码器输出形状: {decoder_output.shape}")  # 应为 [batch_size, tgt_len-1, tgt_vocab_size]
    
    # 测试贪婪解码
    print("\n测试贪婪解码...")
    translated = model.translate(src, max_len=15)
    print(f"翻译结果形状: {translated.shape}")
    print(f"翻译示例: {translated[0, :10]}")
    
    # 测试集束搜索
    print("\n测试集束搜索...")
    beam_translated = model.beam_search(src, beam_size=3, max_len=15)
    print(f"集束搜索结果形状: {beam_translated.shape}")
    print(f"集束搜索示例: {beam_translated[0, :10]}")
    
    # 测试掩码生成
    print("\n测试后续掩码生成...")
    mask = model.generate_square_subsequent_mask(5)
    print("5x5后续掩码:")
    print(mask)
    
    # 模型参数统计
    print("\n模型参数统计:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 各层参数统计
    print("\n各层参数数量:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel():,}")
    
    # 测试模型保存和加载
    print("\n测试模型保存和加载...")
    torch.save(model.state_dict(), "test_transformer_model.pth")
    
    # 创建新模型实例并加载权重
    loaded_model = TransformerNMT(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size
    )
    loaded_model.load_state_dict(torch.load("test_transformer_model.pth"))
    
    # 验证加载的模型输出是否相同
    with torch.no_grad():
        original_output = model(src, tgt[:, :-1])
        loaded_output = loaded_model(src, tgt[:, :-1])
        
        if torch.allclose(original_output, loaded_output, rtol=1e-4):
            print("✓ 模型保存和加载测试通过")
        else:
            print("✗ 模型保存和加载测试失败")
    
    # 清理测试文件
    import os
    if os.path.exists("test_transformer_model.pth"):
        os.remove("test_transformer_model.pth")
    
    # 性能测试
    print("\n性能测试...")
    import time
    
    # 小批量测试
    test_batch_size = 1
    test_src = torch.randint(1, src_vocab_size, (test_batch_size, src_seq_len))
    
    # 贪婪解码性能
    start_time = time.time()
    for _ in range(10):
        _ = model.translate(test_src, max_len=20)
    greedy_time = (time.time() - start_time) / 10
    print(f"贪婪解码平均时间: {greedy_time*1000:.2f}ms")
    
    # 集束搜索性能
    start_time = time.time()
    for _ in range(10):
        _ = model.beam_search(test_src, beam_size=3, max_len=20)
    beam_time = (time.time() - start_time) / 10
    print(f"集束搜索(beam_size=3)平均时间: {beam_time*1000:.2f}ms")
    
    # 不同集束大小比较
    print("\n不同集束大小性能比较:")
    for beam_size in [1, 3, 5, 7]:
        start_time = time.time()
        for _ in range(5):
            _ = model.beam_search(test_src, beam_size=beam_size, max_len=20)
        avg_time = (time.time() - start_time) / 5
        print(f"  beam_size={beam_size}: {avg_time*1000:.2f}ms")
    
    print("\n所有测试完成!")