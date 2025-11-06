import torch
from transformer_enhanced import Transformer, SimpleTokenizer
from config import Config

def generate(model, tokenizer, start_text, max_len=50, temperature=1.0):
    model.eval()
    device = next(model.parameters()).device

    # 初始化序列
    tokens = tokenizer.encode(start_text, max_len=max_len)
    tokens = tokens[:-1]  # 去掉最后一个 pad
    src = torch.tensor([tokens], dtype=torch.long).to(device)

    generated = tokens.copy()

    for _ in range(max_len):
        # 模型前向
        out = model(src, src)  # 简化: 用 src 作为 tgt
        next_token_logits = out[0, -1] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        if next_token == tokenizer.pad_idx:
            break

        generated.append(next_token)
        src = torch.tensor([generated[-len(tokens):]], dtype=torch.long).to(device)

    return tokenizer.decode(generated)

if __name__ == "__main__":
    cfg = Config()
    device = torch.device(cfg.device)

    tokenizer = SimpleTokenizer(vocab_size=cfg.vocab_size, pad_idx=cfg.pad_idx)
    # 加载训练好的模型
    model = Transformer(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
        use_relative=cfg.use_relative,
        use_linear=cfg.use_linear
    ).to(device)
    model.load_state_dict(torch.load(cfg.save_path, map_location=device))

    # 测试生成
    start_text = "Once upon a time"
    generated_text = generate(model, tokenizer, start_text, max_len=50)
    print("Generated:", generated_text)
