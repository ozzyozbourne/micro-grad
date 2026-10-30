# Chapter 20 — tinygrad/llm builds a real transformer out of the same pieces
# Run: DEV=CPU python3 examples/20_llm_tiny.py
#
# tinygrad/llm/model.py's Transformer is made of nn.RMSNorm, nn.Embedding and Linear (a subclass of nn.Linear in
# llm/kernels/amd.py that can also hold quantized GGUF weights). Here we build a *tiny random* one (no download),
# list its state_dict (the same names a GGUF checkpoint uses, which is how from_gguf loads weights with
# load_state_dict), and run a prefill and one decode step on the CPU.
from tinygrad import Tensor, nn, GlobalCounters
from tinygrad.llm.model import Transformer, TransformerConfig

cfg = TransformerConfig(num_blocks=2, dim=32, hidden_dim=64, n_heads=4, n_kv_heads=2, norm_eps=1e-5, vocab_size=50,
                        head_dim=8, rope_theta=10000.0, rope_dim=8, v_head_dim=8, max_context=16)
Tensor.manual_seed(0)
model = Transformer(cfg)
sd = nn.state.get_state_dict(model)
print(f"{len(sd)} tensors, {sum(v.numel() for v in sd.values())} parameters")
for k, v in sd.items():
  if k.startswith("blk.1."): continue
  print(f"  {k:26s} {str(tuple(v.shape)):10s} {type(getattr(model, 'blk')[0]).__name__ if k.startswith('blk') else ''}")
print("  blk.1.* : same 9 tensors as blk.0")
print("layer types:", sorted({type(x).__module__ + "." + type(x).__name__ for x in
      [model.token_embd, model.output_norm, model.output, model.blk[0].attn_q, model.blk[0].ffn_norm]}))

GlobalCounters.reset()
tok = model(Tensor([[1, 2, 3, 4]]), 0, Tensor([0.0]))          # prefill 4 tokens, greedy (temperature 0)
print(f"prefill -> next token {tok.item()}  ({GlobalCounters.kernel_count} kernels)")
GlobalCounters.reset()
tok = model(Tensor([[tok.item()]]), 4, Tensor([0.0]))           # one decode step at position 4
print(f"decode  -> next token {tok.item()}  ({GlobalCounters.kernel_count} kernels)")
