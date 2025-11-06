# data_utils.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter # (确保导入 Counter)
import re

# ==============================
# 简单 Tokenizer (已修复)
# ==============================
class SimpleTokenizer:
    def __init__(self, vocab_size=32000, pad_idx=0, unk_idx=1, sos_idx=2, eos_idx=3, tokenize_mode='auto'):
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        # tokenize_mode: 'auto' | 'word' | 'char'
        self.tokenize_mode = tokenize_mode
        
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_built = False

    def _has_cjk(self, s: str) -> bool:
        # 检测是否包含中文等 CJK 字符
        return re.search(r"[\u4e00-\u9fff]", s) is not None

    def _tokenize(self, text: str):
        s = str(text)
        if self.tokenize_mode == 'char' or (self.tokenize_mode == 'auto' and self._has_cjk(s)):
            # 中文等无空格语言使用字符级分词，保证稳定学习
            return list(s)
        # 英文等使用空格分词
        return s.split()

    def build_vocab(self, texts):
        counter = Counter()
        for txt in texts:
            counter.update(self._tokenize(txt))
        
        # --- [修改] 为 4 个特殊 token 腾出空间 ---
        most_common = counter.most_common(self.vocab_size - 4) 
        
        # (i+4 是因为 0,1,2,3 已经被占用了)
        self.word2idx = {w: i+4 for i, (w, _) in enumerate(most_common)}
        
        self.word2idx['<PAD>'] = self.pad_idx
        self.word2idx['<UNK>'] = self.unk_idx
        self.word2idx['<SOS>'] = self.sos_idx
        self.word2idx['<EOS>'] = self.eos_idx
        # --- [修改结束] ---
        
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_built = True
        
        # 更新真实的 vocab_size
        self.vocab_size = len(self.word2idx)


    def encode(self, text, max_len):
        """
        [核心修复] 
        1. 词元化
        2. 截断 (为 [SOS] 和 [EOS] 留下 2 个位置)
        3. 添加 [SOS] 和 [EOS]
        4. 填充 (到 max_len)
        """
        if not self.vocab_built:
            raise RuntimeError("Vocabulary not built. Call build_vocab() first.")
        
        # 1. 词元化（自动中文字符级分词）
        tokens = self._tokenize(text)
        ids = [self.word2idx.get(tok, self.unk_idx) for tok in tokens]
        
        # 2. 截断
        ids = ids[:max_len - 2]
        
        # 3. 添加 [SOS] 和 [EOS]
        ids = [self.sos_idx] + ids + [self.eos_idx]
        
        # 4. 填充
        padding_len = max_len - len(ids)
        if padding_len > 0:
            ids += [self.pad_idx] * padding_len
            
        return ids

    def decode(self, ids):
        words = []
        for i in ids:
            if i == self.eos_idx: # 遇到 EOS 就停止
                break
            if i == self.pad_idx or i == self.sos_idx: # 忽略 PAD 和 SOS
                continue
            words.append(self.idx2word.get(i, '<UNK>'))
        return ' '.join(words)

# ==============================
# CNN/DailyMail Dataset (无需修改)
# ==============================
class CNNDailyMailDataset(Dataset):
    def __init__(self, articles, summaries, tokenizer, max_src_len, max_tgt_len):
        self.articles = articles
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        # 修复后的 encode 方法会正确处理一切
        src = self.tokenizer.encode(self.articles[idx], self.max_src_len)
        tgt = self.tokenizer.encode(self.summaries[idx], self.max_tgt_len)
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

# ==============================
# DataLoader 构建函数 (已修复)
# ==============================
def get_dataloaders(args, tokenizer):
    # 读取 CSV
    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)

    # 打印列名方便检查
    print("Train columns:", train_df.columns)
    print("Val columns:", val_df.columns)

    # 构建词表 (现在会包含 <SOS> 和 <EOS>)
    tokenizer.build_vocab(train_df['article'].tolist() + train_df['highlights'].tolist())

    # Dataset
    train_dataset = CNNDailyMailDataset(
        train_df['article'].tolist(),
        train_df['highlights'].tolist(),
        tokenizer,
        args.max_src_len,
        args.max_tgt_len
    )
    val_dataset = CNNDailyMailDataset(
        val_df['article'].tolist(),
        val_df['highlights'].tolist(),
        tokenizer,
        args.max_src_len,
        args.max_tgt_len
    )

    # DataLoader
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              num_workers=0) # (保持 num_workers=0)
    
    val_loader = DataLoader(val_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            num_workers=0) # (保持 num_workers=0)

    return train_loader, val_loader