using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;


namespace LLM
{
    
    public partial class MiniLLM : Module<Tensor, Tensor>
    {
        // 一次性返回： layerKey -> (inFeatures, outFeatures)
        // layerKey 已经去掉 .weight，可直接当 _loraDict 的 key
        public Dictionary<string, (long inFeatures, long outFeatures)> ScanLoraTargets()
        {
            var dict = new Dictionary<string, (long, long)>();
            var state = state_dict();

            foreach (var (key, tensor) in state)
            {
                // 只认 .weight 结尾，且是 2-D（Linear 权重）
                if (!key.EndsWith(".weight") || tensor.dim() != 2) continue;

                string layerKey = key[..^7];                // 去掉 ".weight"
                long outFeatures = tensor.shape[0];
                long inFeatures = tensor.shape[1];
                dict[layerKey] = (inFeatures, outFeatures);
            }
            return dict;
        }
    }
}

