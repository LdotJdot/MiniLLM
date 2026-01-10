using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LLM
{
    public partial class MiniLLM : Module<Tensor, Tensor>
    {
        public readonly int _contextLen;
        public readonly int _n_embd;
        public readonly int _n_head;
        public readonly int _n_layer;
        public readonly int _n_kv_head; // 用于 GQA
        public readonly double _rope_base;
        public readonly double resid_dropout = 0;
        public readonly double attn_dropout = 0.0;
        public readonly double ffn_dropout = 0.0;
        public int ContextLen => _contextLen;
        public readonly ModelSize _size;
        public readonly Device _device;
        public readonly int _vocabSize;

        /* ---------- 构造 ---------- */
        public MiniLLM(
            int vocabSize,
            int contextLen,
            Device device,
            ModelSize size,
            int? n_kv_head = null,
            double rope_base = 10000.0
        ) : base("MiniLLM")
        {
            this._vocabSize = vocabSize;
            _contextLen = contextLen;
            this._size = size;
            var cfg = ModelSizeTable.Lookup(this._size);
            _n_layer = cfg.n_layer;
            _n_head = cfg.n_head;
            _n_embd = cfg.n_embd;
            _n_kv_head = n_kv_head ?? cfg.n_head; // 默认不用 GQA
            _rope_base = rope_base;
            this._device = device;

            TokenEmbedding = Embedding(vocabSize, _n_embd);
            // 不再使用 PositionEmbedding，RoPE 在 Attention 里完成

            var blocks = new List<Module<Tensor, Tensor>>();
            for (int i = 0; i < _n_layer; i++)
                blocks.Add(
                    new LlamaBlock(
                        _n_embd,
                        _n_head,
                        _n_kv_head,
                        contextLen,
                        _rope_base,
                        resid_dropout,
                        ffn_dropout,
                        _device
                    )
                );
            // Transformer = Sequential(blocks.ToArray());

            Transformer = new ModuleList<Module<Tensor, Tensor>>(blocks.ToArray()); // lora use

            // 最后输出前再补一个 RMSNorm
            FinalNorm = new RMSNorm(_n_embd);
            LMHead = Linear(_n_embd, vocabSize, hasBias: false);

            // 保证“注册 → to(device) → 初始化”顺序
            RegisterComponents();

            this.to(device);

            InitializeWeights();
        }

        public override Tensor forward(Tensor input)
        {
            var (batchSize, seqLen) = (input.shape[0], input.shape[1]);

            // 1. token 嵌入（无位置嵌入）
            var x = TokenEmbedding.forward(input);

            // 2. Transformer 层

            foreach (var block in Transformer)
            {
                x = block.forward(x);
            }

            // 3. 输出头
            x = FinalNorm.forward(x);
            var logits = LMHead.forward(x);
            return logits;
        }

        /* ---------- 下面是 Llama 风格模块 ---------- */

        // LlamaBlock = Pre-RMSNorm + SwiGLU FFN + RoPE SelfAttention
        internal class LlamaBlock : Module<Tensor, Tensor>
        {
            private readonly RMSNorm _attn_norm;
            private readonly RMSNorm _ffn_norm;
            internal readonly SelfAttentionRoPE _attn;
            internal readonly FeedForwardSwiGLU _ffn;

            private readonly Dropout _residDrop;
            Device _device;

            public LlamaBlock(
                int n_embd,
                int n_head,
                int n_kv_head,
                int max_seq,
                double rope_base,
                double residDropout,
                double ffnDropout,
                Device device
            ) : base("LlamaBlock")
            {
                _device = device;
                _attn_norm = new RMSNorm(n_embd);
                _ffn_norm = new RMSNorm(n_embd);
                _attn = new SelfAttentionRoPE(
                    n_embd,
                    n_head,
                    n_kv_head,
                    max_seq,
                    rope_base,
                    0.0,
                    _device
                );
                _ffn = new FeedForwardSwiGLU(n_embd, ffnDropout);
                _residDrop = Dropout(residDropout);
                RegisterComponents();
            }

            public override Tensor forward(Tensor x)
            {
                // Pre-Norm 结构
                x = x + _residDrop.forward(_attn.forward(_attn_norm.forward(x)));
                x = x + _residDrop.forward(_ffn.forward(_ffn_norm.forward(x)));
                return x;
            }
        }

        // RMSNorm
        public class RMSNorm : Module<Tensor, Tensor>
        {
            private readonly int _dim;
            private readonly Parameter _scale;
            private readonly double _eps;

            public RMSNorm(int dim, double eps = 1e-6) : base("RMSNorm")
            {
                _dim = dim;
                _eps = eps;
                _scale = Parameter(ones(dim));
                RegisterComponents();
            }

            public override Tensor forward(Tensor x)
            {
                if (x.shape[1] == 0) // ← 新增防崩
                    return x;
                // x: (B,T,C)
                var dtype = x.dtype;
                x = x.to(ScalarType.Float32);
                var varx = x.pow(2).mean(new long[] { x.dim() - 1 }, keepdim: true);
                x = x * varx.add_(_eps).rsqrt_();
                return _scale.to(x.dtype) * x.to(dtype);
            }
        }

        // RoPE + optional GQA
        public class SelfAttentionRoPE : Module<Tensor, Tensor>
        {
            private readonly int _n_head;
            private readonly int _n_kv_head;
            private readonly int _head_dim;
            private readonly int _n_rep; // GQA 重复次数
            private readonly Module<Tensor, Tensor> _q_proj,
                _k_proj,
                _v_proj,
                _o_proj; // 其实是Linear，为了后续替代方便
            private readonly Dropout _attn_dropout;
            private readonly RotaryEmbedding _rotary;
            private Tensor? _causal_mask;
            private readonly int _max_seq;
            Device _device;

            public SelfAttentionRoPE(
                int n_embd,
                int n_head,
                int n_kv_head,
                int max_seq,
                double rope_base,
                double dropout,
                Device device
            ) : base("SelfAttentionRoPE")
            {
                this._device = device;
                _max_seq = max_seq;

                _n_head = n_head;
                _n_kv_head = n_kv_head;
                _n_rep = _n_head / _n_kv_head;
                _head_dim = n_embd / n_head;

                _q_proj = Linear(n_embd, n_head * _head_dim, hasBias: false);
                _k_proj = Linear(n_embd, n_kv_head * _head_dim, hasBias: false);
                _v_proj = Linear(n_embd, n_kv_head * _head_dim, hasBias: false);
                _o_proj = Linear(n_head * _head_dim, n_embd, hasBias: false);

                _attn_dropout = Dropout(dropout);
                _rotary = new RotaryEmbedding(_head_dim, max_seq, rope_base, _device);
                RegisterComponents();
                precompute_causal_mask();
            }

            private void precompute_causal_mask()
            {
                _causal_mask = torch
                    .tril(
                        torch.ones(_max_seq, _max_seq, dtype: ScalarType.Float32, device: _device)
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(torch.@bool);
            }

            public override Tensor forward(Tensor x)
            {
                var (B, T, C) = ((int)x.shape[0], (int)x.shape[1], (int)x.shape[2]);

                if (T == 0) // ← 新增防崩
                    return torch.empty(B, 0, C, dtype: x.dtype, device: x.device);

                var q = _q_proj.forward(x).view(B, T, _n_head, _head_dim).transpose(1, 2); // (B,nh,T,hd)
                var k = _k_proj.forward(x).view(B, T, _n_kv_head, _head_dim).transpose(1, 2);
                var v = _v_proj.forward(x).view(B, T, _n_kv_head, _head_dim).transpose(1, 2);

                // RoPE
                q = _rotary.apply(q, T);
                k = _rotary.apply(k, T);

                // GQA 重复 k,v
                if (_n_rep > 1)
                {
                    // 更省内存的重复
                    k = k.unsqueeze(2) // (B, n_kv_h, 1, T, hd)
                        .expand(B, _n_kv_head, _n_rep, T, _head_dim)
                        .reshape(B, _n_kv_head * _n_rep, T, _head_dim);
                    v = v.unsqueeze(2)
                        .expand(B, _n_kv_head, _n_rep, T, _head_dim)
                        .reshape(B, _n_kv_head * _n_rep, T, _head_dim);
                }

                // ===== FlashAttention 路径 =====

                // 不需要手写 mask，is_casual=true 自动完成下三角
                var outTensor = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    p: _attn_dropout.p,
                    is_casual: true
                ); // (B,nh,T,hd)

                outTensor = outTensor.transpose(1, 2).contiguous().view(B, T, C);
                return _o_proj.forward(outTensor);
            }
        }

        // SwiGLU FFN
        internal class FeedForwardSwiGLU : Module<Tensor, Tensor>
        {
            private readonly Module<Tensor, Tensor> _gate_up,
                _down_proj; // 合并 gate 与 up 权重，一次性 matmul
            private readonly Dropout _drop;
            private readonly int _hidden;

            public FeedForwardSwiGLU(int dim, double dropout, int hidden_dim = -1)
                : base("FeedForwardSwiGLU")
            {
                _hidden = hidden_dim > 0 ? hidden_dim : 4 * dim * 2 / 3;
                _hidden = 64 * ((_hidden + 63) / 64); // 64 对齐
                // 合并 gate 与 up：输出 2×hidden，一次性计算
                _gate_up = Linear(dim, 2 * _hidden, hasBias: false);
                _down_proj = Linear(_hidden, dim, hasBias: false);
                _drop = Dropout(dropout);
                RegisterComponents();
            }

            public override Tensor forward(Tensor x)
            {
                if (x.shape[1] == 0) // ← 新增防崩
                    return torch.empty(
                        x.shape[0],
                        0,
                        ((Linear)_down_proj).weight.shape[1],
                        dtype: x.dtype,
                        device: x.device
                    );
                // 单 matmul 得 [gate, up]
                var gate_up = _gate_up.forward(x); // (B,T,2*hidden)

                var half = gate_up.shape[^1] / 2; // 精确一半
                var gate = gate_up[.., .., ..(int)half];
                var up = gate_up[.., .., (int)half..];

                // 就地 silu 与乘，节省一次中间张量
                return _drop.forward(
                    _down_proj.forward(torch.nn.functional.silu(gate, inplace: true) * up)
                );
            }
        }

        // RoPE 实现（简化版，支持 Llama 默认 10000 基频）
        private class RotaryEmbedding : Module
        {
            private readonly int _dim;
            private readonly int _max_seq;
            private readonly double _base;
            private Tensor? _cos,
                _sin;
            private Tensor? _cos_buf; // (max_seq, head_dim)
            private Tensor? _sin_buf;
            Device _device;

            public RotaryEmbedding(int dim, int max_seq, double @base, Device device)
                : base("RotaryEmbedding")
            {
                _dim = dim;
                _max_seq = max_seq;
                _base = @base;
                this._device = device;

                RegisterComponents(); // 把 _cos_buf / _sin_buf 注册进去
                precompute();
            }

            private void precompute()
            {
                var inv_freq =
                    1.0
                    / torch.pow(
                        _base,
                        torch.arange(0, _dim, 2, dtype: ScalarType.Float32, device: _device) / _dim
                    );
                var t = torch
                    .arange(_max_seq, dtype: ScalarType.Float32, device: _device)
                    .unsqueeze(1);
                var f = t * inv_freq.unsqueeze(0); // (max_seq, dim/2)

                // 一次性算好 cos/sin 并注册为 buffer
                // 原来这里用了两次 cos(f) 拼接成 32，现在只拼一次，保持 dim==head_dim
                _cos_buf = torch.cos(f); // (max_seq, dim/2)
                _sin_buf = torch.sin(f); // (max_seq, dim/2)
            }

            public Tensor apply(Tensor x, int seqLen)
            {
                // x: (B,nh,T,hd)
                var cos = _cos_buf!.to(x.dtype)[..seqLen, ..].unsqueeze(0).unsqueeze(0); // (1,1,T,hd/2)
                var sin = _sin_buf!.to(x.dtype)[..seqLen, ..].unsqueeze(0).unsqueeze(0); // (1,1,T,hd/2)
                return rotate_half(x, cos, sin);
            }

            private static Tensor rotate_half(Tensor x, Tensor cos, Tensor sin)
            {
                var halfDim = x.shape[^1] / 2;
                var x1 = x[.., .., .., ..(int)halfDim];
                var x2 = x[.., .., .., (int)halfDim..];

                var rot_x1 = x1 * cos - x2 * sin;
                var rot_x2 = x1 * sin + x2 * cos;

                return torch.cat(new[] { rot_x1, rot_x2 }, dim: -1);
            }
        }

        /* ---------- 原有公共接口保持不变 ---------- */
        public Embedding TokenEmbedding { get; }

        // public Sequential Transformer { get; }
        public ModuleList<Module<Tensor, Tensor>> Transformer { get; } // lora use
        public RMSNorm FinalNorm { get; }
        public Linear LMHead { get; }

        public void ClipGradNorm(double maxNorm = 1.0)
        {
            // 把所有可训练参数展平后一次性裁剪
            var allParams = parameters().Where(p => p.requires_grad);
            torch.nn.utils.clip_grad_norm_(allParams, maxNorm);
        }

        public void InitializeWeights()
        {
            foreach (var (name, p) in named_parameters())
            {
                if (p.dim() < 2)
                    continue;

                if (name.Contains("o_proj") || name.Contains("down_proj"))
                {
                    // 输出投影用 zero-init 或极小值，保持初始残差为 0
                    torch.nn.init.zeros_(p);
                }
                else if (
                    name.Contains("q_proj")
                    || name.Contains("k_proj")
                    || name.Contains("v_proj")
                    || name.Contains("gate_up")
                )
                {
                    // 输入投影用 1/sqrt(fan_in)
                    var fanIn = p.shape[1];
                    torch.nn.init.normal_(p, mean: 0, std: 1.0 / Math.Sqrt(fanIn));
                }
                else if (name.Contains("embedding"))
                {
                    torch.nn.init.normal_(p, 0, 0.02);
                }
            }
        }

        public static MiniLLM? LoadModel(
            string filepath,
            int vocabSize,
            int contextLen,
            Device device,
            ModelSize size
        )
        {
            try
            {
                var model = new MiniLLM(vocabSize, contextLen, device, size);
                model.load(filepath);
                model.to(device).eval();
                Console.WriteLine($"模型已从 {filepath} 加载");
                return model;
            }
            catch (Exception ex)
            {
                Console.WriteLine(
                    $"加载模型失败: {ex.Message}, vocabSize:{vocabSize},contextLen:{contextLen},size:{size}"
                );
                return null;
            }
        }

        public void SaveModel(string filepath)
        {
            try
            {
                train();
                save(filepath);
                Console.WriteLine($"模型已保存到 {filepath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"保存模型失败: {ex.Message}");
            }
        }
    }
}
