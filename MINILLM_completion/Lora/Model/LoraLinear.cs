// LoRA 公式： y = Wx + BAx ，其中 B,A 为可训练低秩矩阵，W 冻结
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Lora.Model
{
    public class LoraLinear : Module<Tensor, Tensor>
    {
        internal readonly Linear _frozen;   // 原始权重，frozen
        internal readonly Linear _loraA;    // r×in_features
        internal readonly Linear _loraB;    // out_features×r
        internal readonly double _alpha;  // alpha/r
        internal readonly string  _baseKey;     // ← 新增：base 权重在 state_dict 里的名字

    
        public LoraLinear(Linear original, int rank, double alpha,Device device, string baseKey, bool mergeWeights = false)
            : base($"LoraLinear_{baseKey}")
        {
            _baseKey = baseKey;
            _frozen = original;
            _frozen.eval();                 // 保险起见
            foreach (var p in _frozen.parameters())
                p.requires_grad = false;    // 冻结主干

            long outDim = original.weight.shape[0];
            long inDim = original.weight.shape[1];

            _loraA = Linear(inDim, rank, hasBias: false);
            _loraB = Linear(rank, outDim, hasBias: false);

            // 搬运到指定设备
            _loraA.to(device);
            _loraB.to(device);

            _alpha = alpha / rank;
            // 初始化：A 正态，B 全零 → 训练前输出不变
            torch.nn.init.kaiming_normal_(_loraA.weight);
            torch.nn.init.zeros_(_loraB.weight);

            // 1. 把权重包成 Parameter（默认 requires_grad=true）
            //_loraA.weight = new Parameter(_loraA.weight);
            //_loraB.weight = new Parameter(_loraB.weight);

            _loraA.weight.requires_grad = true;
            _loraB.weight.requires_grad = true;

            RegisterComponents();  // 把 _loraA/_loraB 登记到子模块
            if (mergeWeights) Merge(); // 若直接推理，可立即合并
        }

        public override Tensor forward(Tensor x)
        {
            var main = _frozen.forward(x);
            var lora = _loraB.forward(_loraA.forward(x)) * _alpha;
            return main + lora;
        }

        /// <summary>
        /// 把 LoRA 权重合并到主干，之后可去掉 LoRA 部分，速度+显存双收益
        /// </summary>
        public void Merge()
        {

            using (torch.no_grad())
            {
                var delta = _loraB.weight.mm(_loraA.weight).mul_(_alpha); // out×in
                _frozen.weight.add_(delta);   // ← 原地加
            }
        }

        /// <summary>
        /// 取消合并（如果之前 Merge 过）
        /// </summary>
        public void UnMerge()
        {
            using (torch.no_grad())
            {
                // 1. 先算低秩矩阵  B·A
                var delta = _loraB.weight.mm(_loraA.weight);   // out×r  @  r×in  →  out×in
                                                               // 2. 再乘缩放系数
                delta *= _alpha;                               // _alpha == alpha/r
                                                               // 3. 从主干减掉
                _frozen.weight.sub_(delta);
            }
        }
    }
}
