using LLM;
using static TorchSharp.torch;



namespace Lora.Model
{
    public partial class LoraMiniLLM
    {

        /// <summary>
        /// 将 LoRA 增量永久合并到原始 Linear 权重，并返回一个**纯 MiniLLM** 对象。
        /// 合并后就可以扔掉 LoRA，只保存/加载普通 minillm 权重。
        /// </summary>
        public MiniLLM MergeAndReturnBaseModel()
        {
            var baseState = _core.state_dict();
            var mergedState = baseState.ToDictionary(kv => kv.Key, kv => kv.Value.clone());

            foreach (var (realKey, lora) in _loraDict)
            {
                Tensor W = baseState[realKey];     // 现在一定存在
                Tensor delta = lora._loraB.weight.matmul(lora._loraA.weight) * lora._alpha;
                mergedState[realKey] = W + delta;
            }

            var newModel = new MiniLLM(_core._vocabSize,
                                       _core.ContextLen,
                                       _core._device,
                                       _core._size,
                                       _core._n_kv_head,
                                       _core._rope_base);
            newModel.load_state_dict(mergedState);
            return newModel;
        }
        /// <summary>
        /// 一步到位：合并 + 保存成普通 minillm 检查点
        /// </summary>
        public void SaveMergedModel(string path)
        {
            var mergedModel = MergeAndReturnBaseModel();
            mergedModel.SaveModel(path);          // 用你原来的 save 方法即可
            Console.WriteLine($"[Save] 已合并 LoRA 并保存为普通 MiniLLM：{path}");
        }
    }
}