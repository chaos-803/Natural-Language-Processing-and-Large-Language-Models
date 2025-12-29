"""
基于RNN的神经机器翻译模型实现
包含编码器、解码器、注意力机制和集束搜索
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List, Dict, Any

class EncoderRNN(nn.Module):
    """RNN编码器，使用双向LSTM/GRU"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 512, 
                 hidden_size: int = 512, num_layers: int = 2, 
                 dropout: float = 0.3, bidirectional: bool = True,
                 cell_type: str = "lstm"):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.cell_type = cell_type.lower()
        self.num_directions = 2 if bidirectional else 1
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout_layer = nn.Dropout(dropout)
        
        # RNN层
        if cell_type == "lstm":
            self.rnn = nn.LSTM(
                embedding_dim, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif cell_type == "gru":
            self.rnn = nn.GRU(
                embedding_dim, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:
            raise ValueError(f"不支持的RNN类型: {cell_type}，请选择 'lstm' 或 'gru'")
        
        # 将双向输出合并为单隐藏层维度
        if bidirectional:
            self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
            self.fc_cell = nn.Linear(hidden_size * 2, hidden_size) if cell_type == "lstm" else None
    
    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        前向传播
        
        参数:
            src: 源语言序列 [batch_size, seq_len]
            src_lengths: 源序列实际长度 [batch_size]
            
        返回:
            outputs: 编码器所有时刻的输出 [batch_size, seq_len, hidden_size * num_directions]
            hidden: 最终隐藏状态
        """
        batch_size = src.size(0)
        
        # 词嵌入
        embedded = self.embedding(src)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout_layer(embedded)
        
        # 打包序列以提高效率
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # 通过RNN
        packed_outputs, hidden = self.rnn(packed_embedded)
        
        # 解包序列
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        
        # 如果是双向RNN，合并前向和后向的隐藏状态
        if self.bidirectional:
            if self.cell_type == "lstm":
                # 处理LSTM的隐藏状态和细胞状态
                h_n, c_n = hidden
                # 合并双向隐藏状态
                h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_size)
                h_n = torch.cat([h_n[:, 0, :, :], h_n[:, 1, :, :]], dim=2)  # [num_layers, batch_size, hidden_size*2]
                h_n = self.fc_hidden(h_n)  # [num_layers, batch_size, hidden_size]
                
                # 合并双向细胞状态
                c_n = c_n.view(self.num_layers, 2, batch_size, self.hidden_size)
                c_n = torch.cat([c_n[:, 0, :, :], c_n[:, 1, :, :]], dim=2)
                c_n = self.fc_cell(c_n)
                
                hidden = (h_n, c_n)
                
                # 合并双向输出
                outputs = self.fc_hidden(outputs)
            else:
                # 处理GRU的隐藏状态
                h_n = hidden
                h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_size)
                h_n = torch.cat([h_n[:, 0, :, :], h_n[:, 1, :, :]], dim=2)
                h_n = self.fc_hidden(h_n)
                hidden = h_n
                
                # 合并双向输出
                outputs = self.fc_hidden(outputs)
        
        return outputs, hidden


class Attention(nn.Module):
    """注意力机制模块，支持多种评分函数"""
    
    def __init__(self, hidden_size: int, method: str = "dot"):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        
        if method == "general":
            self.attn = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == "concat":
            self.attn = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self.v = nn.Parameter(torch.rand(hidden_size))
        elif method == "bahdanau":
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)
            self.U = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v = nn.Parameter(torch.rand(hidden_size))
        elif method != "dot":
            raise ValueError(f"不支持的注意力机制: {method}，请选择 'dot', 'general', 'concat', 或 'bahdanau'")
    
    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算注意力权重
        
        参数:
            hidden: 解码器隐藏状态 [batch_size, hidden_size] 或 [batch_size, num_layers, hidden_size]
            encoder_outputs: 编码器输出 [batch_size, seq_len, hidden_size]
            mask: 注意力掩码 [batch_size, seq_len]
            
        返回:
            attention_weights: 注意力权重 [batch_size, seq_len]
            context_vector: 上下文向量 [batch_size, hidden_size]
        """
        if hidden.dim() == 3:
            # 使用最后一层隐藏状态
            hidden = hidden[-1]  # [batch_size, hidden_size]
        
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        
        # 根据不同的注意力机制计算分数
        if self.method == "dot":
            # 点积注意力: hidden * encoder_outputs
            hidden_expanded = hidden.unsqueeze(1)  # [batch_size, 1, hidden_size]
            scores = torch.bmm(hidden_expanded, encoder_outputs.transpose(1, 2))  # [batch_size, 1, src_len]
            scores = scores.squeeze(1)  # [batch_size, src_len]
            
        elif self.method == "general":
            # 通用注意力: hidden * W * encoder_outputs
            encoder_transformed = self.attn(encoder_outputs)  # [batch_size, src_len, hidden_size]
            hidden_expanded = hidden.unsqueeze(1)  # [batch_size, 1, hidden_size]
            scores = torch.bmm(hidden_expanded, encoder_transformed.transpose(1, 2))  # [batch_size, 1, src_len]
            scores = scores.squeeze(1)
            
        elif self.method == "concat":
            # 连接注意力: v^T * tanh(W[h; encoder_outputs])
            hidden_expanded = hidden.unsqueeze(1).expand(-1, src_len, -1)  # [batch_size, src_len, hidden_size]
            combined = torch.cat((hidden_expanded, encoder_outputs), dim=2)  # [batch_size, src_len, hidden_size*2]
            energy = torch.tanh(self.attn(combined))  # [batch_size, src_len, hidden_size]
            scores = torch.matmul(energy, self.v)  # [batch_size, src_len]
            
        elif self.method == "bahdanau":
            # Bahdanau注意力: v^T * tanh(W*hidden + U*encoder_outputs)
            hidden_processed = self.W(hidden).unsqueeze(1)  # [batch_size, 1, hidden_size]
            encoder_processed = self.U(encoder_outputs)  # [batch_size, src_len, hidden_size]
            energy = torch.tanh(hidden_processed + encoder_processed)  # [batch_size, src_len, hidden_size]
            scores = torch.matmul(energy, self.v)  # [batch_size, src_len]
        
        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=1)  # [batch_size, src_len]
        
        # 计算上下文向量
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # [batch_size, 1, hidden_size]
        context_vector = context_vector.squeeze(1)  # [batch_size, hidden_size]
        
        return attention_weights, context_vector


class DecoderRNN(nn.Module):
    """基于注意力的RNN解码器"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 512, 
                 hidden_size: int = 512, num_layers: int = 2,
                 dropout: float = 0.3, attention_method: str = "dot"):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout_layer = nn.Dropout(dropout)
        
        # 注意力机制
        self.attention = Attention(hidden_size, attention_method)
        
        # GRU解码器
        self.gru = nn.GRU(
            embedding_dim + hidden_size,  # 输入: 词嵌入 + 上下文向量
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 输出层
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size * 2 + embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size)
        )
    
    def forward(self, input_token: torch.Tensor, hidden: torch.Tensor, 
                encoder_outputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        单步解码
        
        参数:
            input_token: 当前输入词 [batch_size, 1]
            hidden: 解码器隐藏状态 [num_layers, batch_size, hidden_size] 或 (h_n, c_n) for LSTM
            encoder_outputs: 编码器输出 [batch_size, src_len, hidden_size]
            mask: 注意力掩码 [batch_size, src_len]
            
        返回:
            output: 输出词的概率分布 [batch_size, vocab_size]
            hidden: 更新后的隐藏状态
            attention_weights: 注意力权重 [batch_size, src_len]
        """
        batch_size = input_token.size(0)
        
        # 词嵌入
        embedded = self.embedding(input_token)  # [batch_size, 1, embedding_dim]
        embedded = self.dropout_layer(embedded)
        
        # 获取当前隐藏状态（如果是LSTM，只使用隐藏状态，不使用细胞状态）
        if isinstance(hidden, tuple):
            # LSTM: hidden是元组 (h_n, c_n)
            current_hidden = hidden[0][-1]  # 使用最后一层的隐藏状态
        else:
            # GRU: hidden是张量
            current_hidden = hidden[-1]  # 使用最后一层的隐藏状态
        
        # 计算注意力
        attention_weights, context_vector = self.attention(current_hidden, encoder_outputs, mask)
        
        # 将词嵌入和上下文向量连接作为GRU输入
        gru_input = torch.cat((embedded, context_vector.unsqueeze(1)), dim=2)  # [batch_size, 1, embedding_dim + hidden_size]
        
        # GRU前向传播
        output, hidden = self.gru(gru_input, hidden if not isinstance(hidden, tuple) else hidden[0])
        
        # 如果是LSTM，需要保持细胞状态不变
        if isinstance(hidden, tuple):
            # 更新隐藏状态，但保持细胞状态
            h_n, c_n = hidden
            hidden = (output.transpose(0, 1), c_n)
            output = output.squeeze(1)  # [batch_size, hidden_size]
        else:
            output = output.squeeze(1)  # [batch_size, hidden_size]
        
        # 准备最终输出
        output = torch.cat((output, context_vector, embedded.squeeze(1)), dim=1)  # [batch_size, hidden_size*2 + embedding_dim]
        output = self.fc_out(output)  # [batch_size, vocab_size]
        output = F.log_softmax(output, dim=1)
        
        return output, hidden, attention_weights


class RNNNMT(nn.Module):
    """基于RNN的神经机器翻译模型"""
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, 
                 embedding_dim: int = 512, hidden_size: int = 512,
                 num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = True, cell_type: str = "lstm",
                 attention_method: str = "dot"):
        super(RNNNMT, self).__init__()
        
        # 编码器
        self.encoder = EncoderRNN(
            vocab_size=src_vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            cell_type=cell_type
        )
        
        # 解码器
        self.decoder = DecoderRNN(
            vocab_size=tgt_vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            attention_method=attention_method
        )
        
        # 模型参数
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        
        # 特殊标记索引
        self.pad_idx = 0
        self.sos_idx = 2
        self.eos_idx = 3
    
    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor, 
                tgt: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        训练模式的前向传播
        
        参数:
            src: 源语言序列 [batch_size, src_len]
            src_lengths: 源序列实际长度 [batch_size]
            tgt: 目标语言序列 [batch_size, tgt_len]
            teacher_forcing_ratio: 教师强制比率
            
        返回:
            outputs: 解码器所有时刻的输出 [batch_size, tgt_len-1, tgt_vocab_size]
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        
        # 编码器前向传播
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # 准备输出张量
        outputs = torch.zeros(batch_size, tgt_len - 1, self.tgt_vocab_size, device=src.device)
        
        # 解码器第一个输入是<SOS>标记
        decoder_input = tgt[:, 0].unsqueeze(1)  # [batch_size, 1]
        
        # 创建注意力掩码
        src_mask = self.create_mask(src, self.pad_idx)
        
        # 解码循环
        for t in range(1, tgt_len):
            decoder_output, hidden, _ = self.decoder(
                decoder_input, hidden, encoder_outputs, src_mask
            )
            
            outputs[:, t-1, :] = decoder_output
            
            # 决定下一个输入是教师强制还是模型预测
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            
            if use_teacher_forcing:
                # 使用真实标签作为下一个输入
                decoder_input = tgt[:, t].unsqueeze(1)
            else:
                # 使用模型预测作为下一个输入
                _, topi = decoder_output.topk(1)
                decoder_input = topi.detach()
        
        return outputs
    
    def create_mask(self, src: torch.Tensor, pad_idx: int) -> torch.Tensor:
        """创建注意力掩码"""
        mask = (src != pad_idx).unsqueeze(1)  # [batch_size, 1, src_len]
        return mask
    
    def translate(self, src: torch.Tensor, src_lengths: torch.Tensor, 
                  max_len: int = 100) -> torch.Tensor:
        """
        贪婪解码翻译
        
        参数:
            src: 源语言序列 [batch_size, src_len]
            src_lengths: 源序列实际长度 [batch_size]
            max_len: 最大生成长度
            
        返回:
            translated_ids: 翻译结果 [batch_size, seq_len]
        """
        batch_size = src.size(0)
        
        # 编码器前向传播
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # 解码器第一个输入是<SOS>标记
        decoder_input = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=src.device)
        
        # 存储翻译结果
        translated_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=src.device)
        translated_ids[:, 0] = self.sos_idx
        
        # 创建注意力掩码
        src_mask = self.create_mask(src, self.pad_idx)
        
        # 解码循环
        for t in range(1, max_len):
            decoder_output, hidden, _ = self.decoder(
                decoder_input, hidden, encoder_outputs, src_mask
            )
            
            # 获取最可能的词
            _, topi = decoder_output.topk(1)
            decoder_input = topi.detach()
            
            # 存储预测结果
            translated_ids[:, t] = topi.squeeze(1)
            
            # 如果所有序列都生成了<EOS>，则停止
            if (decoder_input == self.eos_idx).all():
                break
        
        return translated_ids
    
    def beam_search(self, src: torch.Tensor, src_lengths: torch.Tensor, 
                    beam_size: int = 5, max_len: int = 100) -> torch.Tensor:
        """
        集束搜索解码
        
        参数:
            src: 源语言序列 [batch_size, src_len]
            src_lengths: 源序列实际长度 [batch_size]
            beam_size: 集束大小
            max_len: 最大生成长度
            
        返回:
            best_sequence: 最佳翻译序列 [batch_size, seq_len]
        """
        batch_size = src.size(0)
        
        # 编码器前向传播
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # 创建注意力掩码
        src_mask = self.create_mask(src, self.pad_idx)
        
        # 初始化解码器输入
        start_token = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=src.device)
        
        # 存储每个批次的最终结果
        all_best_sequences = []
        
        # 对每个样本单独进行集束搜索
        for i in range(batch_size):
            # 提取单个样本的编码器输出和隐藏状态
            encoder_output = encoder_outputs[i:i+1].expand(beam_size, -1, -1)
            if isinstance(hidden, tuple):
                # LSTM
                h_n, c_n = hidden
                hidden_i = (h_n[:, i:i+1].expand(-1, beam_size, -1),
                           c_n[:, i:i+1].expand(-1, beam_size, -1))
            else:
                # GRU
                hidden_i = hidden[:, i:i+1].expand(-1, beam_size, -1)
            
            src_mask_i = src_mask[i:i+1].expand(beam_size, -1, -1)
            
            # 初始化集束
            beams = [{
                'sequence': [self.sos_idx],
                'score': 0.0,
                'hidden': hidden_i,
                'decoder_input': torch.tensor([[self.sos_idx]], device=src.device)
            }]
            
            completed_beams = []
            
            for step in range(max_len):
                candidates = []
                
                for beam in beams:
                    # 如果已经生成EOS，则不再扩展
                    if beam['sequence'][-1] == self.eos_idx:
                        candidates.append(beam)
                        continue
                    
                    # 解码下一步
                    decoder_output, new_hidden, _ = self.decoder(
                        beam['decoder_input'],
                        beam['hidden'],
                        encoder_output,
                        src_mask_i
                    )
                    
                    # 获取top-k个候选
                    log_probs = decoder_output.squeeze(0)  # [vocab_size]
                    topk_probs, topk_indices = log_probs.topk(beam_size)
                    
                    for j in range(beam_size):
                        token_id = topk_indices[j].item()
                        token_score = topk_probs[j].item()
                        
                        candidate = {
                            'sequence': beam['sequence'] + [token_id],
                            'score': beam['score'] + token_score,
                            'hidden': new_hidden,
                            'decoder_input': torch.tensor([[token_id]], device=src.device)
                        }
                        candidates.append(candidate)
                
                # 按分数排序并选择top beam_size个
                candidates.sort(key=lambda x: x['score'] / len(x['sequence']), reverse=True)
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
                all_beams.sort(key=lambda x: x['score'] / len(x['sequence']), reverse=True)
                best_sequence = all_beams[0]['sequence']
            
            # 填充或截断序列
            if len(best_sequence) < max_len:
                best_sequence = best_sequence + [self.pad_idx] * (max_len - len(best_sequence))
            else:
                best_sequence = best_sequence[:max_len]
            
            all_best_sequences.append(best_sequence)
        
        # 转换为张量
        return torch.tensor(all_best_sequences, device=src.device)


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
    embedding_dim = 512
    hidden_size = 512
    num_layers = 2
    
    # 创建模型实例
    model = RNNNMT(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.3,
        bidirectional=True,
        cell_type="lstm",
        attention_method="dot"
    )
    
    # 创建随机输入
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    src_lengths = torch.randint(src_seq_len//2, src_seq_len+1, (batch_size,))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    # 测试前向传播
    print("测试前向传播...")
    outputs = model(src, src_lengths, tgt, teacher_forcing_ratio=0.5)
    print(f"输出形状: {outputs.shape}")  # 应为 [batch_size, tgt_len-1, tgt_vocab_size]
    print(f"输出值范围: [{outputs.min():.4f}, {outputs.max():.4f}]")
    
    # 测试贪婪解码
    print("\n测试贪婪解码...")
    translated = model.translate(src, src_lengths, max_len=15)
    print(f"翻译结果形状: {translated.shape}")
    print(f"翻译示例: {translated[0, :10]}")
    
    # 测试集束搜索
    print("\n测试集束搜索...")
    beam_translated = model.beam_search(src, src_lengths, beam_size=3, max_len=15)
    print(f"集束搜索结果形状: {beam_translated.shape}")
    print(f"集束搜索示例: {beam_translated[0, :10]}")
    
    # 测试不同配置
    print("\n测试不同RNN单元类型...")
    for cell_type in ["lstm", "gru"]:
        model_gru = RNNNMT(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            cell_type=cell_type
        )
        outputs = model_gru(src, src_lengths, tgt)
        print(f"{cell_type.upper()} 前向传播成功，输出形状: {outputs.shape}")
    
    print("\n测试不同注意力机制...")
    for attn_method in ["dot", "general", "concat", "bahdanau"]:
        model_attn = RNNNMT(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            attention_method=attn_method
        )
        outputs = model_attn(src, src_lengths, tgt)
        print(f"{attn_method}注意力 前向传播成功，输出形状: {outputs.shape}")
    
    # 模型参数统计
    print("\n模型参数统计:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 测试模型保存和加载
    print("\n测试模型保存和加载...")
    torch.save(model.state_dict(), "test_rnn_model.pth")
    
    # 创建新模型实例并加载权重
    loaded_model = RNNNMT(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size
    )
    loaded_model.load_state_dict(torch.load("test_rnn_model.pth"))
    
    # 验证加载的模型输出是否相同
    with torch.no_grad():
        original_output = model(src, src_lengths, tgt)
        loaded_output = loaded_model(src, src_lengths, tgt)
        
        if torch.allclose(original_output, loaded_output, rtol=1e-4):
            print("✓ 模型保存和加载测试通过")
        else:
            print("✗ 模型保存和加载测试失败")
    
    # 清理测试文件
    import os
    if os.path.exists("test_rnn_model.pth"):
        os.remove("test_rnn_model.pth")
    
    print("\n所有测试完成!")