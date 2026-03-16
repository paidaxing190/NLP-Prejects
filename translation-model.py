from torch import nn
import torch
import config  
from tokenizer import ChineseTokenizer,EnglishTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        # 正弦/余弦位置编码实现
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # 不参与梯度更新

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
class TransformerTranslationModel(nn.Module):
    def __init__(self, zh_vocab_size, en_vocab_size, zh_pad_idx, en_pad_idx):
        super().__init__()
        # 词嵌入层
        self.src_emb = nn.Embedding(zh_vocab_size, config.EMBEDDING_DIM, padding_idx=zh_pad_idx)
        self.tgt_emb = nn.Embedding(en_vocab_size, config.EMBEDDING_DIM, padding_idx=en_pad_idx)
        
        # 位置编码层
        self.pos_encoder = PositionalEncoding(config.EMBEDDING_DIM, config.DROPOUT)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=config.EMBEDDING_DIM,
            nhead=config.NHEAD,          
            num_encoder_layers=config.NUM_LAYERS,
            num_decoder_layers=config.NUM_LAYERS,
            dim_feedforward=config.HIDDEN_SIZE,
            dropout=config.DROPOUT,
            batch_first=True,            
            device=device                
        )
        
        # 输出层：映射到目标语言词汇表
        self.fc_out = nn.Linear(config.EMBEDDING_DIM, en_vocab_size)
        
        self.top_k = config.TOP_K if hasattr(config, 'TOP_K') else 5

    def forward(self, src, tgt):
        """
        前向传播：中文→英文翻译
        Args:
            src: 中文输入序列 [batch_size, src_seq_len]
            tgt: 英文目标序列 [batch_size, tgt_seq_len]
        Returns:
            output: 英文token预测分布 [batch_size, tgt_seq_len, en_vocab_size]
        """
        # 嵌入+位置编码
        src_emb = self.pos_encoder(self.src_emb(src))  # [batch, src_len, emb_dim]
        tgt_emb = self.pos_encoder(self.tgt_emb(tgt))  # [batch, tgt_len, emb_dim]
        
        # 生成掩码（避免解码器看到未来token）
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)
        # Padding掩码（忽略padding token的注意力计算）
        src_pad_mask = (src == self.src_emb.padding_idx).to(device)
        
        # Transformer前向计算
        output = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask
        )
        
        # 映射到词汇表
        output = self.fc_out(output)
        return output

    def top_k_sampling(self, logits):
        top_k_vals, top_k_indices = torch.topk(logits, self.top_k)
        probs = nn.functional.softmax(top_k_vals, dim=-1)
        selected_idx = torch.multinomial(probs, 1)
        return top_k_indices[selected_idx]

def count_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_trained_model(model_path, zh_vocab_size, en_vocab_size, zh_pad_idx, en_pad_idx):
    model = TransformerTranslationModel(zh_vocab_size, en_vocab_size, zh_pad_idx, en_pad_idx).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

if __name__ == "__main__":

    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR/'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR/'en_vocab.txt')
    # 初始化模型
    model = TransformerTranslationModel(zh_tokenizer.vocab_size, en_tokenizer.vocab_size, zh_tokenizer.pad_token_index, en_tokenizer.pad_token_index).to(device)
  
    src = torch.randint(1, zh_tokenizer.vocab_size, (config.BATCH_SIZE, config.MAX_SEQ_LEN)).to(device)
    tgt = torch.randint(1, en_tokenizer.vocab_size, (config.BATCH_SIZE, config.MAX_SEQ_LEN)).to(device)
    
    # 前向传播测试
    output = model(src, tgt)
    print(f"模型输出形状: {output.shape}")  
    print(f"模型参数量: {count_model_params(model) / 1e6:.2f}M")  
