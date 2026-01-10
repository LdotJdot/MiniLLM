using MINILLM_Completion.Tokenizers;
using System;
using System.Collections.Generic;
using System.Text;
using static TorchSharp.torch;

public interface IDatasetGenerator
{
    public int Size { get; }
    public Task<(Tensor inputs, Tensor targets)> GetBatch();
    public ITokenizer Tokenizer { get; }
}
