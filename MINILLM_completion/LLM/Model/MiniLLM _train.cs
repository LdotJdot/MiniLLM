using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;


namespace LLM
{
    
    public partial class MiniLLM : Module<Tensor, Tensor>
    {

        /// <summary>
        /// 训练入口（带梯度累加、fp32-loss、Nan/Inf 保护、epoch/batch 级 checkpoint）
        /// </summary>
        /// <param name="dataset">必须实现 GetBatch() 返回 (inputs, targets) 且已 to(_device)</param>
        /// <param name="numEpochs">总 epoch 数</param>
        /// <param name="learningRate">AdamW 初始 lr</param>
        /// <param name="batchSize">单卡 batch-size（未考虑分布式）</param>
        /// <param name="verbose">每 N 个 batch 打印一次日志</param>
        /// <param name="checkPointSaveAfterEpochNum">每 N 个 epoch 保存一次 checkpoint；0=不保存</param>
        /// <param name="checkPointSaveAfterEpochPeriod">是否在 verbose 时机额外保存 checkpoint</param>
        /// <param name="promptForTestAfterEpochPeriodText">打印日志时同步做生成的 prompt；空=不生成</param>
        /// <param name="checkPointModel">checkpoint 文件名</param>
        public async Task Train(
            IDatasetGenerator dataset,
            int numEpochs,
            float learningRate,
            int batchSize,
            int verbose = 20,
            int checkPointSaveAfterEpochNum = 0,
            bool checkPointSaveAfterEpochPeriod = true,
            string promptForTestAfterEpochPeriodText = "你好！",
            string checkPointModel = "minillm_model_epoch_checkPoint.pt")
        {
            #region 1. 初始化优化器与损失函数
            var _params = parameters().ToList();
            var optimizer = torch.optim.AdamW(_params, lr: learningRate, weight_decay: 0.1);
            var criterion = torch.nn.CrossEntropyLoss();
            const int gradAccum = 2;                 // 梯度累加步数,两个梯度累加后再更新参数
            int stepsPerEpoch = (dataset.Size + batchSize - 1) / batchSize;
            Console.WriteLine("开始训练...");
            Console.WriteLine($"数据集共 {dataset.Size} 条样本，每 epoch {stepsPerEpoch} 个 step，batchSize={batchSize}");
            #endregion

            // 2. 主循环
            for (int epoch = 0; epoch < numEpochs; epoch++)
            {
                float totalLoss = 0f;   // 累加“真实”loss 总和（未除 gradAccum）
                int successfulBatches = 0;
                int accumCount = 0;     // 当前已累加 batch 数

                // 2.1 训练模式（只需一次）
                train();

                for (int batchIdx = 0; batchIdx < stepsPerEpoch; batchIdx++)
                {
                    try
                    {
                        /* ===== 2.1 取 batch ===== */
                        var (inputs, targets) = await dataset.GetBatch();
                        inputs = inputs.to(_device);
                        targets = targets.to(_device);

                        /* ===== 2.2 前向 + 损失（fp32） ===== */
                        using var _ = torch.NewDisposeScope();   // 自动释放中间 Tensor
                        using var outputs = forward(inputs);     // 模型内部保持 fp16

                        using var logits = outputs.to(ScalarType.Float32)   // 关键：转 fp32 再算 loss
                                                 .reshape(-1, outputs.shape[2]);
                        using var targets_reshaped = targets.reshape(-1);

                        using var loss = criterion.forward(logits, targets_reshaped); // fp32 计算


                        if (float.IsNaN(loss.item<float>()) || float.IsInfinity(loss.item<float>()))
                        {
                            Console.WriteLine($"Epoch {epoch}, Batch {batchIdx}: 损失为 NaN/Inf，跳过");
                            optimizer.zero_grad();   // 清空已累加梯度
                            accumCount = 0;
                            continue;
                        }

                        // 先累加“真实”loss，再缩放后反向
                        totalLoss += loss.item<float>();
                        (loss / gradAccum).backward();
                        successfulBatches++;
                        accumCount++;

                        /* ===== 2.3 梯度累加更新 ===== */
                        if (accumCount == gradAccum || batchIdx == stepsPerEpoch - 1)
                        {
                            ClipGradNorm(1.0);   // 梯度裁剪
                            optimizer.step();    // 真正更新参数
                            optimizer.zero_grad(); // 清空梯度
                            accumCount = 0;
                        }

                        /* ===== 2.4 日志 / 采样 ===== */
                        if (verbose > 0 && (batchIdx + 1) % verbose == 0)
                        {
                            float avgLoss = totalLoss / successfulBatches; // 平均loss（totalLoss已经是原始值）
                                                        
                            double ppl = Math.Exp(avgLoss);          // <-- 这就是 PPL

                            Console.WriteLine($"Epoch {epoch + 1}/{numEpochs}, " +
                                            $"Step {batchIdx + 1}/{stepsPerEpoch}, " +
                                            $"Loss: {avgLoss:F4}, " +
                                            $"PPL: {ppl:F2}");

                            // 按需保存 checkpoint
                            if (checkPointSaveAfterEpochPeriod)
                            {
                                Console.WriteLine($"Save checkpoint at batch {batchIdx + 1}/{stepsPerEpoch}");
                                SaveModel(checkPointModel);
                            }

                            // 按需生成文本
                            if (!string.IsNullOrWhiteSpace(promptForTestAfterEpochPeriodText))
                            {
                                string generated = GenerateTextFromPrompt(
                                                   promptForTestAfterEpochPeriodText,
                                                   dataset.Tokenizer, maxTokens: 100);
                                Console.WriteLine("生成样例：\n" + generated);
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Epoch {epoch}, Batch {batchIdx} 训练失败: {ex.Message}");
                        optimizer.zero_grad(); // 保证异常后梯度不污染
                        accumCount = 0;
                    }
                }

                /* ===== 2.5 结束一个 epoch ===== */
                if (successfulBatches > 0)
                {
                    float avgLoss = totalLoss / successfulBatches;
                    Console.WriteLine($"\nEpoch {epoch + 1}/{numEpochs} 完成，平均损失: {avgLoss:F4}\n");
                }

                // 按 epoch 保存 checkpoint
                if (checkPointSaveAfterEpochNum > 0 &&
                    (epoch + 1) % checkPointSaveAfterEpochNum == 0)
                {
                    Console.WriteLine($"Save checkpoint at epoch {epoch + 1}");
                    SaveModel(checkPointModel);
                }
            }
        }
    }
}

