from torch import nn
import config
import torch
from tokenizer import JiebaTokenizer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True 

class BertStyleDynamicEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx):
        super().__init__()
        
        self.static_emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        # BERT思路：上下文感知变换层（动态调整词向量）
        self.context_linear = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)  
        self.dropout = nn.Dropout(config.DROPOUT if hasattr(config, 'DROPOUT') else 0.1)

    def forward(self, x):
        # 静态词向量
        static_emb = self.static_emb(x)  # [batch_size, seq_len, embedding_dim]
        # 上下文变换
        dynamic_emb = self.context_linear(static_emb)
        # 残差连接+LayerNorm
        emb = self.layer_norm(static_emb + dynamic_emb)
        return self.dropout(emb)

class ReviewAnalyzeModel(nn.Module):
    def __init__(self, vocab_size, padding_index):
        super().__init__()
        self.embedding = BertStyleDynamicEmbedding(vocab_size, config.EMBEDDING_DIM, padding_index)
        
        self.lstm = nn.LSTM(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS if hasattr(config, 'NUM_LAYERS') else 1,
            bidirectional=True, 
            batch_first=True,
            dropout=config.DROPOUT if (hasattr(config, 'DROPOUT') and config.NUM_LAYERS > 1) else 0
        )
        
        # 适配双向LSTM的输出维度（hidden_size*2）
        self.linear = nn.Linear(config.HIDDEN_SIZE * 2, 1)
        
       
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x):
        # x.shape: [batch_size, seq_len]
        embed = self.embedding(x)  
        # embed.shape: [batch_size, seq_len, embedding_dim]
        
        output, (_, _) = self.lstm(embed)
        # output.shape: [batch_size, seq_len, hidden_size*2]（双向LSTM）
        
       
        batch_indexes = torch.arange(0, output.shape[0])
        lengths = (x != self.embedding.static_emb.padding_idx).sum(dim=1)
        last_hidden = output[batch_indexes, lengths - 1]
        # last_hidden.shape: [batch_size, hidden_size*2]
        
        output = self.linear(last_hidden).squeeze(dim=-1)
        # output.shape: [batch_size]
        return output

def count_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_trained_model(model_path, vocab_size, padding_index):
    model = ReviewAnalyzeModel(vocab_size, padding_index).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def calculate_accuracy(preds, labels, threshold=0.0):
    preds = (torch.sigmoid(preds) > threshold).float()
    correct = (preds == labels).sum().item()
    return correct / len(labels)

if __name__ == "__main__":
   
    tokenizer = JiebaTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    # 初始化模型并移到显卡
    model = ReviewAnalyzeModel(tokenizer.vocab_size, tokenizer.pad_token_index).to(device)
    
   
    batch_size = config.BATCH_SIZE if hasattr(config, 'BATCH_SIZE') else 32
    max_seq_len = config.MAX_SEQ_LEN if hasattr(config, 'MAX_SEQ_LEN') else 50
    test_input = torch.randint(1, tokenizer.vocab_size, (batch_size, max_seq_len)).to(device)
    test_label = torch.randint(0, 2, (batch_size,)).float().to(device)
    
    # 前向传播测试
    output = model(test_input)
    acc = calculate_accuracy(output, test_label)
    
    print(f"模型输出形状: {output.shape}")  
    print(f"模型参数量: {count_model_params(model) / 1e6:.2f}M")
    print(f"模拟准确率: {acc:.2f}")  
