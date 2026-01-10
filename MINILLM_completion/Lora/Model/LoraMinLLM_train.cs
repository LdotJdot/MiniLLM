using MINILLM_Completion.GlobalSetting;
using MINILLM_Completion.Lora.DataLoader;
using TorchSharp;
using static TorchSharp.torch;

namespace Lora.Model;

public partial class LoraMiniLLM
{
    /// <summary>
    /// LoRA 微调入口（仅训练 LoRA 参数）
    /// </summary>
    public async Task Train(
        TextDataSet_Lora dataset,          // ← 唯一变化：换成 SFT 数据集
        int numEpochs,
        float learningRate,
        int batchSize,
        int verbose = 20,
        int checkPointSaveAfterEpochNum = 0,
        bool checkPointSaveAfterEpochPeriod = true,
        string promptForTestAfterEpochPeriodText = "你好！",
        string checkPointModel = "minillm_model_epoch_checkPoint.sft",
        string mergecheckPointModel = "loraminillm_epoch_checkPoint.sft")

    {
        #region 1. 只把 LoRA 参数交给优化器
        var loraParams = parameters()
                        .Where(p => p.requires_grad)   // 前面冻基座时只有 LoRA 张量需要梯度
                        .ToList();
        var optimizer = torch.optim.AdamW(loraParams, lr: learningRate, weight_decay: 0.1);
        var criterion = torch.nn.CrossEntropyLoss(ignore_index: Symbol.MASK); // SFT 已把 source 置 -100
        const int gradAccum = 2;
        int stepsPerEpoch = (dataset.Size + batchSize - 1) / batchSize;
        Console.WriteLine("========== LoRA 训练 ==========");
        Console.WriteLine($"数据集 {dataset.Size} 条，每 epoch {stepsPerEpoch} step，batchSize={batchSize}");
        Console.WriteLine($"可训练张量数: {loraParams.Count}");
        foreach (var p in loraParams)
            Console.WriteLine($" {p.name}  -  {string.Join(',', p.shape)}");
        #endregion

        for (int epoch = 0; epoch < numEpochs; epoch++)
        {
            float totalLoss = 0f;
            int goodBatch = 0;
            int accumCnt = 0;

            train();                       // 只需一次

            for (int batchIdx = 0; batchIdx < stepsPerEpoch; batchIdx++)
            {
                try
                {
                    /* ===== 取 batch ===== */
                    var (inputs, targets) = await dataset.GetBatch(); // 已 to(device)
                    inputs = inputs.to(_device);
                    targets = targets.to(_device);
                    using var _ = torch.NewDisposeScope();

                    /* ===== 前向 + loss（fp32） ===== */
                    using var outputs = forward(inputs);               // 模型内可保持 fp16
                    using var logits = outputs.to(ScalarType.Float32)
                                              .reshape(-1, outputs.shape[2]);
                    using var loss = criterion.forward(logits, targets.reshape(-1));

                    if (float.IsNaN(loss.item<float>()) || float.IsInfinity(loss.item<float>()))
                    {
                        Console.WriteLine($"Epoch {epoch} Batch {batchIdx}: NaN/Inf，跳过");
                        optimizer.zero_grad(); accumCnt = 0; continue;
                    }

                    totalLoss += loss.item<float>();
                    (loss / gradAccum).backward();
                    goodBatch++;
                    accumCnt++;

                    /* ===== 梯度累加更新 ===== */
                    if (accumCnt == gradAccum || batchIdx == stepsPerEpoch - 1)
                    {
                        ClipGradNorm(1.0);   // 仅裁剪 LoRA 梯度
                        optimizer.step();
                        optimizer.zero_grad();
                        accumCnt = 0;
                    }

                    /* ===== 日志 / 采样 ===== */
                    if (verbose > 0 && (batchIdx + 1) % verbose == 0)
                    {
                        float avgLoss = totalLoss / goodBatch;
                        double ppl = Math.Exp(avgLoss);
                        Console.WriteLine(
                            $"Epoch {epoch + 1}/{numEpochs}  " +
                            $"Step {batchIdx + 1}/{stepsPerEpoch}  " +
                            $"Loss {avgLoss:F4}  PPL {ppl:F2}");

                        if (checkPointSaveAfterEpochPeriod)
                        {
                            SaveModelLora(checkPointModel);
                            SaveMergedModel(mergecheckPointModel);
                        }

                        if (!string.IsNullOrWhiteSpace(promptForTestAfterEpochPeriodText))
                        {
                            string outText = GenerateTextFromPrompt(promptForTestAfterEpochPeriodText, dataset.Tokenizer, maxTokens: 100);
                            Console.WriteLine(outText);
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Epoch {epoch} Batch {batchIdx} 异常：{ex.Message}");
                    optimizer.zero_grad(); accumCnt = 0;
                }
            }

            /* ===== epoch 结束 ===== */
            if (goodBatch > 0)
                Console.WriteLine(
                    $"Epoch {epoch + 1}/{numEpochs} 完成，" +
                    $"平均 Loss {totalLoss / goodBatch:F4}");

            if (checkPointSaveAfterEpochNum > 0 &&
                (epoch + 1) % checkPointSaveAfterEpochNum == 0)
            {
                SaveModelLora(checkPointModel);
                Console.WriteLine("Lora checkpoint saved at batch period");
                SaveMergedModel(mergecheckPointModel);
                Console.WriteLine("MergedModel checkpoint saved at batch period");
            }
        }
    }
}