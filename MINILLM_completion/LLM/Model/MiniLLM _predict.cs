using MINILLM_Completion.Tokenizers;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;


namespace LLM
{
    
    public partial class MiniLLM : Module<Tensor, Tensor>
    {


        /// <summary>
        /// 主生成接口
        /// </summary>
        public string GenerateTextFromPrompt(string prompt,
                                             ITokenizer tokenizer,
                                             int maxTokens = 500,
                                             bool includeSpecial = true,
                                             float temperature = 0.3f,
                                             float topP = 0.6f,
                                             float repetitionPenalty = 1.15f)
        {
            eval();                       // 你的模型切换到 eval 模式
            using (torch.no_grad())
            {
                var generated = tokenizer.Encode(prompt).ToList();

                var generatedSet = new HashSet<int>(generated); // 用于重复惩罚
                var resultToken = new List<int>();
                for (int i = 0; i < maxTokens; i++)
                {
                    // 只保留最后 ContextLen 个 token
                    var inputTokens = generated.Count > ContextLen
                                      ? generated.Skip(generated.Count - ContextLen).ToList()
                                      : generated;

                    var inputTensor = tensor(inputTokens.ToArray(), dtype: torch.int64)
                                        .unsqueeze(0)
                                        .to(_device);

                    var logits = forward(inputTensor);        // 你的 forward 返回 [1, seq, vocab]
                    var nextToken = SampleNextToken(logits[0, -1],
                                                   temperature,
                                                   topP,
                                                   repetitionPenalty,
                                                   generatedSet);

                    generated.Add(nextToken);
                    generatedSet.Add(nextToken);
                    resultToken.Add(nextToken);
                    if (tokenizer.IsEndToken(nextToken))
                        break;
                }

                return tokenizer.Decode(resultToken, includeSpecial);
            }
        }

        /// <summary>
        /// Top-p + 重复惩罚 采样核心
        /// </summary>
        private int SampleNextToken(Tensor logits,          // shape: [vocab]
                                   float temperature,
                                   float topP,
                                   float repetitionPenalty,
                                   HashSet<int> generatedTokens)
        {
            // 0. 重复惩罚（对已出现 token 降权）
            if (generatedTokens?.Count > 0)
            {
                foreach (var idx in generatedTokens)
                {
                    var old = logits[idx].ToSingle();
                    // 标准做法：正值除以penalty，负值乘以penalty
                    logits[idx] = old < 0 ? old * repetitionPenalty : old / repetitionPenalty;
                }
            }

            // 1. 温度缩放（完整 logits）
            if (temperature != 1.0f)
                logits = logits / temperature;

            // 2. 计算概率并做 Top-p 截断
            var (sortedProbs, sortedIndices) = logits.softmax(dim: -1).sort(descending: true);
            var cumProbs = sortedProbs.cumsum(dim: -1);

            // 找到第一个超过topP的位置（至少保留1个token）
            var lastPos = (cumProbs <= topP).sum().to(torch.CPU).item<long>();
            if (lastPos < 1) lastPos = 1;  // 确保至少有一个token可选

            var probs = sortedProbs[..(int)lastPos];
            var indices = sortedIndices[..(int)lastPos];

            // 3. multinomial 采样
            var sample = torch.multinomial(probs, num_samples: 1);
            var vocabIdx = indices[sample.to(torch.CPU)].item<long>();

            return (int)vocabIdx;
        }
    }

}

