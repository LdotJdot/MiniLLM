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


public abstract class DatasetGenerator_Encoded: IDatasetGenerator
{
    protected readonly List<List<int>> _texts;
    protected readonly Random _random;
    protected readonly int _seqLen;

    protected int batchSize;
    public int Size => _texts?.Count ?? 0;
    protected Device device;
    protected ITokenizer _tokenizer;
    public ITokenizer Tokenizer => _tokenizer;
    public int VocabSize => _tokenizer.GetVocabSize();

    protected DataCacheAsync<(Tensor inputs, Tensor targets)>? _dataCache;
    protected EpochCircularSampler<int> sampler;
    public int workerNum { get; }

    public DatasetGenerator_Encoded(Device device, ITokenizer tokenizer, List<List<int>> texts, int batchSize, int workerNum, int seqLen, int? seed = null)
    {
        this.device = device;
        this.batchSize = batchSize;
        this.workerNum = workerNum;

        _tokenizer = tokenizer;

        _texts = texts;
        _seqLen = seqLen;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
        this.workerNum = workerNum;

        // 初始化索引并打乱
        sampler = new EpochCircularSampler<int>(Enumerable.Range(0, _texts.Count));


        _dataCache = new DataCacheAsync<(Tensor inputs, Tensor targets)>(batchSize * 100);

        _dataCache.StartProducer(GenerateBatch, workerNum);
    }

    public async Task<(Tensor inputs, Tensor targets)> GetBatch()
    {
        return await _dataCache!.Consume();
    }


    /// <summary>
    /// 获取一个 batch，支持 epoch 概念。
    /// 当所有样本用完后会自动 reshuffle。
    /// </summary>
    protected abstract (Tensor inputs, Tensor targets) GenerateBatch();

    protected abstract (int[] input, int[] target) PrepareSequence(ReadOnlySpan<int> tokens, int seqLen);

    protected (Tensor inputs, Tensor targets) CreateTensors(
      List<int[]> inputs,
      List<int[]> targets,
      int batchSize)
    {
        const int bytesPerInt = sizeof(int);
        int seqLen = inputs[0].Length;          // 64
        int totalElements = inputs.Count * seqLen;   // 704
        int totalBytes = totalElements * bytesPerInt; // 2816

        int[] bufIn = ArrayPool<int>.Shared.Rent(totalElements);
        int[] bufTar = ArrayPool<int>.Shared.Rent(totalElements);

        try
        {
            // 元素级偏移，但 BlockCopy 用字节
            for (int i = 0; i < inputs.Count; i++)
            {
                int byteOffset = i * seqLen * bytesPerInt;
                Buffer.BlockCopy(inputs[i], 0, bufIn, byteOffset, seqLen * bytesPerInt);
                Buffer.BlockCopy(targets[i], 0, bufTar, byteOffset, seqLen * bytesPerInt);
            }


            // 按数组池的传入，长度可能超出 batch 需求，需 narrow
            var tIn = tensor(bufIn, ScalarType.Int64)
                      .narrow(0, 0, totalElements)
                      .reshape(batchSize, _seqLen)
                      .detach();
            var tTar = tensor(bufTar, ScalarType.Int64)
                           .narrow(0, 0, totalElements)
                           .reshape(batchSize, _seqLen)
                           .detach();

            return (tIn.detach(), tTar.detach());
        }
        finally
        {
            ArrayPool<int>.Shared.Return(bufIn);
            ArrayPool<int>.Shared.Return(bufTar);
        }
    }

}




