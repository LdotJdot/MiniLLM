using LLM;
using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using TorchSharp;
using TorchSharp.Modules;
using static LLM.MiniLLM;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Lora.Model
{
    /// <summary>
    /// 把已有 MiniLLM 包装成「可训练 LoRA」版本，对外接口保持一致
    /// </summary>
    public partial class LoraMiniLLM : Module<Tensor, Tensor>
    {
        private readonly MiniLLM _core;
        private readonly Dictionary<string, LoraLinear> _loraDict = new();
        private readonly Dictionary<string, (Tensor weight, string stateDictKey)> _weight2Key = new();
        private Device _device => _core._device;
        public LoraMiniLLM(MiniLLM core,
                           int loraRank = 16,
                           double loraAlpha = 32,
                           string[]? targetModules = null) : base("LoraMiniLLM")
        {
            _core = core;
            // 默认只给 attention 和 FFN 的 q/k/v/o/up/down 加 LoRA
            targetModules ??= new[]
            {
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_up", "down_proj"
            };

            ReplaceWithLora(targetModules, loraRank, loraAlpha);
            this.to(_device);
            RegisterComponents(); // 登记所有 LoraLinear
        }

        /* ---------- 前向完全复用 core ---------- */
        public override Tensor forward(Tensor input) => _core.forward(input);

        /* ---------- 常用辅助 ---------- */
        public new void train(bool mode = true) { _core.train(mode); base.train(mode); }
        public new void eval() { _core.eval(); base.eval(); }

        public void SaveModelLora(string path)
        {
            var dict = new Dictionary<string, Tensor>();
            foreach (var (key, lora) in _loraDict)
            {
                dict[$"{key}.loraA"] = lora._loraA.weight;
                dict[$"{key}.loraB"] = lora._loraB.weight;
            }
            TensorPak.Save(dict, path);
            Console.WriteLine($"[Save] LoRA 权重已保存，共 {dict.Count} 个张量");
        }

        public void LoadLoraOnly(string path)
        {
            var dict = TensorPak.Load(path);
            foreach (var (key, lora) in _loraDict)
            {
                if (dict.TryGetValue($"{key}.loraA", out var ta))
                    lora._loraA.weight = new Parameter(ta.to(_device));
                if (dict.TryGetValue($"{key}.loraB", out var tb))
                    lora._loraB.weight = new Parameter(tb.to(_device));
            }
            Console.WriteLine($"[Load] LoRA 权重已加载");
        }

        private void ReplaceWithLora(string[] targets, int r, double alpha)
        {
            for (int i = 0; i < _core.Transformer.Count; i++)
            {
                var block = (MiniLLM.LlamaBlock)_core.Transformer[i];
                // attention
                ReplaceModuleLinears(i, block._attn, targets, r, alpha);
                // ffn
                ReplaceModuleLinears(i, block._ffn, targets, r, alpha);
            }
        }


        private void ReplaceModuleLinears(int blockIdx,
                                  Module<Tensor, Tensor> module,
                                  string[] targets,
                                  int r,
                                  double alpha)
        {
            const BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance;

            var baseState = _core.state_dict();          // 只在这一处拿，避免多次反射

            foreach (var f in module.GetType().GetFields(flags))
            {
                if (!targets.Any(t => f.Name.EndsWith(t))) continue;
                if (f.GetValue(module) is not Linear baseLin) continue;

                /* ==== 先拿到 backing-field 的真实 key ==== */
                string realKey = baseState
                                 .First(kv => kv.Value.Handle == baseLin.weight.Handle)
                                 .Key;                      // 例如 <Transformer>k__BackingField.0._attn._q_proj.weight

                /* ==== 缓存起来，合并时直接用它 ==== */
                _weight2Key[realKey] = (baseLin.weight, realKey);

                var lora = new LoraLinear(baseLin, r, alpha, _device, realKey);
                _loraDict[realKey] = lora;
                f.SetValue(module, lora);

                Console.WriteLine($"[LoRA] 替换 {realKey}");
            }
        }

        /* ---------- 合并/解除合并 ---------- */
        public void MergeLora() { foreach (var l in _loraDict.Values) l.Merge(); }
        public void UnMergeLora() { foreach (var l in _loraDict.Values) l.UnMerge(); }

       
        private const BindingFlags AllFlags =
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance;

        public void ClipGradNorm(double maxNorm = 1.0)
        {
            // 把所有可训练参数展平后一次性裁剪
            var allParams = parameters().Where(p => p.requires_grad);
            torch.nn.utils.clip_grad_norm_(allParams, maxNorm);
        }


    }
}


file static class TensorPak
{
    private record Chunk(string Key, long[] Shape, ScalarType Dtype, long Offset, long Numel);

    /* 把字典保存成单个文件 */
    public static void Save(IReadOnlyDictionary<string, Tensor> dict, string path)
    {
        var chunks = new List<Chunk>();
        long offset = 0;

        // 1. 全部 flatten → float32，拼到一个大 buffer
        using var pw = new BinaryWriter(File.Create(path + ".bin"));
        foreach (var kv in dict)
        {
            var t = kv.Value.cpu().detach().to(ScalarType.Float32);
            long n = t.numel();
            chunks.Add(new Chunk(kv.Key, t.shape, kv.Value.dtype, offset, n));
            offset += n;

            // 按 float32 写出
            var span = t.data<float>().ToArray();
            foreach (var v in span)
                pw.Write(v);
        }
        pw.Flush();

        // 2. 头信息写 JSON
        File.WriteAllText(path + ".json", JsonSerializer.Serialize(chunks));
    }

    /* 从文件读回字典 */
    public static Dictionary<string, Tensor> Load(string path)
    {
        var chunks = JsonSerializer.Deserialize<Chunk[]>(
                        File.ReadAllText(path + ".json"))!;

        // 一次性把大 buffer 读回
        using var pr = new BinaryReader(File.OpenRead(path + ".bin"));
        var total = chunks.Sum(c => c.Numel);
        var buf = new float[total];
        for (long i = 0; i < total; i++)
            buf[i] = pr.ReadSingle();

        var dict = new Dictionary<string, Tensor>();
        long idx = 0;
        foreach (var c in chunks)
        {
            var t = tensor(buf.Skip((int)idx).Take((int)c.Numel).ToArray(),
                          dtype: ScalarType.Float32)
                         .reshape(c.Shape)
                         .to(c.Dtype);
            dict[c.Key] = t;
            idx += c.Numel;
        }
        return dict;
    }
}