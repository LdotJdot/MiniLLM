using LLM;
using MINILLM_Completion.GlobalSetting;
using MINILLM_Completion.Tokenizers;
using MINILLM_Completion.Utils.DataLoader;
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using static TorchSharp.torch;



public class TextDataSet_Pretrain:DatasetGenerator
{
    public TextDataSet_Pretrain(Device device, ITokenizer tokenizer, IList<string> texts, int batchSize, int workerNum, int seqLen, int? seed = null) : base(device, tokenizer, texts, batchSize, workerNum, seqLen, seed)
    {
    }


    /// <summary>
    /// 获取一个 batch，支持 epoch 概念。
    /// 当所有样本用完后会自动 reshuffle。
    /// </summary>
    protected override (Tensor inputs, Tensor targets) GenerateBatch()
    {
        var inputs = new List<int[]>(batchSize);
        var targets = new List<int[]>(batchSize);

       // var sw = System.Diagnostics.Stopwatch.StartNew();


        var batch = sampler.TakeBatch(batchSize);

        foreach(var index in batch)
        {
            var sentence = _texts[index];
            var tokens = _tokenizer.Encode(sentence);
           
            // 截取或填充到固定长度
            var (inputSeq, targetSeq) = PrepareSequence(CollectionsMarshal.AsSpan(tokens), _seqLen);

            inputs.Add(inputSeq);
            targets.Add(targetSeq);
        }            

        // Console.WriteLine($"{_currentIndex} /{_indices.Length}");
        return CreateTensors(inputs, targets, batchSize);
    }   
  
    protected  (int[] input, int[] target) PrepareSequence(
    ReadOnlySpan<int> tokens, int seqLen)
    {
        Debug.Assert(seqLen > 0);
        Debug.Assert(tokens.Length > 0, "tokens不应该为空");

        int[] input, target;

        if (tokens.Length < seqLen + 1)          // padding 分支
        {
            int len = Math.Max(0, tokens.Length - 1);  // 防止tokens.Length=0时len=-1
            input = GC.AllocateUninitializedArray<int>(seqLen);
            target = GC.AllocateUninitializedArray<int>(seqLen);

            tokens.Slice(0, len).CopyTo(input);      // 一次拷贝
            tokens.Slice(1, len).CopyTo(target);

            input.AsSpan(len).Clear();               // 一次清零
            target.AsSpan(len).Clear();
        }
        else                                     // 裁剪分支
        {
            int start = _random.Next(tokens.Length - seqLen);

            input = tokens.Slice(start, seqLen).ToArray();
            target = tokens.Slice(start + 1, seqLen).ToArray();
            // 上面两行可以用下面“二”里的 SIMD 加速拷
        }

        return (input, target);
    }
}




