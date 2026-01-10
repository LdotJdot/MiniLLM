using MINILLM_Completion.Tokenizers;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;


namespace Lora.Model;

public partial class LoraMiniLLM
{
    /// <summary>
    /// 主生成接口
    /// </summary>
    public string GenerateTextFromPrompt(string prompt,
                                         ITokenizer tokenizer,
                                         int maxTokens = 100,
                                         float temperature = 0.7f,
                                         float topP = 0.92f,
                                         float repetitionPenalty = 1.15f)
    {
        eval();                       // 你的模型切换到 eval 模式
        using (torch.no_grad())
        {
            var generated = tokenizer.Encode(prompt).ToList();
            var generatedSet = new HashSet<int>(generated); // 用于重复惩罚

            for (int i = 0; i < maxTokens; i++)
            {
                // 只保留最后 ContextLen 个 token
                var inputTokens = generated.Count > _core.ContextLen
                                  ? generated.Skip(generated.Count - _core.ContextLen).ToList()
                                  : generated;

                var inputTensor = tensor(inputTokens.ToArray(), dtype: torch.int64)
                                    .unsqueeze(0)
                                    .to(_core._device);

                var logits = forward(inputTensor);        // 你的 forward 返回 [1, seq, vocab]
                var nextToken = SampleNextToken(logits[0, -1],
                                               temperature,
                                               topP,
                                               repetitionPenalty,
                                               generatedSet);

                generated.Add(nextToken);
                generatedSet.Add(nextToken);

                if (tokenizer.IsEndToken(nextToken))
                    break;
            }

            return tokenizer.Decode(generated);
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
                logits[idx] = old / repetitionPenalty;
            }
        }

        // 1. 温度缩放（完整 logits）
        if (temperature != 1.0f)
            logits = logits / temperature;

        // 2. 计算概率并做 Top-p 截断
        var (sortedProbs, sortedIndices) = logits.softmax(dim: -1).sort(descending: true);
        var cumProbs = sortedProbs.cumsum(dim: -1);

        // 找到最后一个 <= topP 的位置
        var lastPos = (cumProbs <= topP).sum().to(torch.CPU).item<long>() - 1;
        if (lastPos < 0) lastPos = 0;

        var probs = sortedProbs[..((int)lastPos + 1)];
        var indices = sortedIndices[..((int)lastPos + 1)];

        // 3. multinomial 采样
        var sample = torch.multinomial(probs, num_samples: 1);
        var vocabIdx = indices[sample.to(torch.CPU)].item<long>();

        return (int)vocabIdx;
    }
}


