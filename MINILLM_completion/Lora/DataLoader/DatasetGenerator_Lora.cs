using LLM;
using MINILLM_Completion.GlobalSetting;
using MINILLM_Completion.Tokenizers;
using MINILLM_Completion.Utils.DataLoader;
using System;
using System.Buffers;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.distributions.constraints;


public class TextDataSet_Lora:DatasetGenerator
{
    public TextDataSet_Lora(Device device, ITokenizer tokenizer, IList<string> texts, int batchSize, int workerNum, int seqLen, int? seed = null) : base(device, tokenizer, texts, batchSize, workerNum, seqLen, seed)
    {
    }

    /// <summary>
    /// 为 LoRA 预训练（CLM）生成一个 batch
    /// </summary>
    protected override (Tensor inputs, Tensor targets) GenerateBatch()
    {
        var inputIds = new List<int[]>(batchSize);
        var labels = new List<int[]>(batchSize);

        // 1. 采样 batch 条索引
        foreach (var idx in sampler.TakeBatch(batchSize))
        {
            var sentence = _texts[idx];
            var tokens = _tokenizer.Encode(sentence);   // List<int>

            var (inp, lbl) = MakeSftPair(tokens);

            // 后面再走你原来的 PrepareSequence / 截断 / 填充
            var (paddedInp, paddedLbl) = PrepareSequence(CollectionsMarshal.AsSpan(tokens), lbl.AsSpan(), _seqLen);

            inputIds.Add(paddedInp);
            labels.Add(paddedLbl);
        }

        // 3. List -> Tensor
        return CreateTensors(inputIds, labels, batchSize);
    }

    // 固定的 magic number，就是你已经测出来的 token
    static readonly int[] userStart = { 1, 320, 275 };               //硬编码，等价于 <|im_start|>user
    static readonly int[] assistantStart = { 1, 1078, 538, 501 };    //硬编码， <|im_start|>assistant
    (int[] inputIds, int[] labels) MakeSftPair(List<int> fullTokens)
    {
        int len = fullTokens.Count;
        int[] inputIds = fullTokens.ToArray();
        int[] labels = new int[len];

        // 1. 全部 mask
        Array.Fill(labels, Symbol.MASK);

        var assistantStartSpan = assistantStart.AsSpan();
        var aspLength = assistantStartSpan.Length;
        var inputIdsSpan = inputIds.AsSpan();
        var labelsSpan = labels.AsSpan();

        // 2. 扫描 assistant 区间
        for (int i = 1; i < len;)
        {
            // 探测 <|im_start|>assistant
            if (i + aspLength <= len && assistantStartSpan.SequenceEqual(inputIdsSpan.Slice(i, aspLength)))
            {
                int segStart = i;

                // 一直扫到、并包含结束标记 imend
                while (i < len && !this.Tokenizer.IsEndToken(inputIds[i])) i++;

                // 多个结束标记 eos 
                while (i < len && this.Tokenizer.IsEndToken(inputIds[i])) i++;

                var segLen = i - segStart;

                inputIdsSpan.Slice(segStart, segLen).CopyTo(labelsSpan.Slice(segStart - 1, segLen));
            }
            else
                i++;
        }
        return (inputIds, labels);
    }

    protected (int[] input, int[] target) PrepareSequence(
    ReadOnlySpan<int> inputs,ReadOnlySpan<int> labels, int seqLen)
    {
        Debug.Assert(seqLen > 0);
        Debug.Assert(inputs.Length == labels.Length);
        Debug.Assert(inputs.Length > 0, "inputs不应该为空");

        int[] input;
        int[] target;

        if (inputs.Length < seqLen + 1)          // padding 分支
        {
            int len = Math.Max(0, inputs.Length - 1);  // 防止inputs.Length=0时len=-1
            input = GC.AllocateUninitializedArray<int>(seqLen);
            target = GC.AllocateUninitializedArray<int>(seqLen);

            inputs.Slice(0, len).CopyTo(input);      // 一次拷贝
            labels.Slice(0, len).CopyTo(target);      // 一次拷贝

            input.AsSpan(len).Clear();               // 一次清零
            target.AsSpan(len).Clear();               // 一次清零
        }
        else                                     // 裁剪分支
        {
            int start = _random.Next(inputs.Length - seqLen);

            input = inputs.Slice(start, seqLen).ToArray();
            target = labels.Slice(start, seqLen).ToArray();
        }

        return (input, target);
    }
}




