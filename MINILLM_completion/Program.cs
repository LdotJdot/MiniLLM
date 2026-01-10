using LLM;
using Lora.Model;
using MINILLM_Completion.LLM.Config;
using MINILLM_Completion.Lora.DataLoader;
using MINILLM_Completion.Tokenizers;
using MINILLM_Completion.Utils;
using System.Text.RegularExpressions;
using TorchSharp;
using static System.Net.Mime.MediaTypeNames;

class Program
{  
    static async Task Main(string[] args)
    {
        LogDualWriter.Attach();   // 控制台输出附加到log

        LLMConfig config = LLMConfig.Load(PathHelper.ConfigPath);

        char option = (char)0;
        while (true)
        {
            Console.WriteLine("键入1训练，键入2训练Lora，键入0预测");
            option = Console.ReadKey(true).KeyChar;
            if (option == '1')
            {
                Console.WriteLine("In train mode.");

                await Train(config);
            }
            else if (option == '2')
            {
                Console.WriteLine("In lora mode.");
                await TrainLora(config);
            }
            else if (option == '0')
            {
                Console.WriteLine("In predict mode.");
                Predict(config);
            }
            else
            {
                Console.WriteLine("无效选项，请重试");
            }
        }
        
    }

    static async Task Train(LLMConfig config)
    {
        // 配置
        Console.WriteLine(config.ToString());

        int contextLen = config.contextLen;    //128
        int batchSize = config.batchSize;
        int seqLen = config.seqLen;
        int numEpochs = config.numEpochs;
        float learningRate = config.learningRate;
        int verbose = config.verbose;
        int checkPointSaveAfterEpochNum = config.checkPointSaveAfterEpochNum;
        bool checkPointSaveAfterEpochPeriod = config.checkPointSaveAfterEpochPeriod;
        string promptForTestAfterEpochPeriodText = config.promptForTestAfterEpochPeriodText;
        string loadCurrentModel = config.loadCurrentModel;
        string checkPointModel = config.checkPointModel;
        string saveCurrentModel = config.saveCurrentModel;
        int workerNumber = config.workerNumber ?? 3;  
        ModelSize modelSize = (ModelSize)Enum.Parse(typeof(ModelSize), config.modelSize, ignoreCase: true);

        Console.WriteLine("配置加载完成");

        // 处理Dataload时CPU片时占用高
        var data = JsonLReader_Pretrain.ConvertJsonlToIntArrayArray(config.pretrainPath);  //如果训练数据是tokenizer预处理好的token数组，直接加载，在运行时无需encode

        // 处理Dataload时CPU片时占用低
        // var data = JsonLReader_Pretrain.ConvertJsonlToStringArray(config.pretrainPath);  //如果训练数据是原始的jsonl格式，直接反序列化加载，用tokenizer在运行时encode

        try
        {

            // 1. 设置设备
            var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            Console.WriteLine($"Using device: {device}");
            var seed = 42;
            torch.random.manual_seed(seed);

            var tokenizer = new Tokenizer(config.tokenizerPath);
            Console.WriteLine($"Tokenizer initialized.");

            // 创建数据集（包含tokenizer）
            var dataset = new DatasetGenerator_Pretrain_Encoded(
                device,
                tokenizer,
                data,
                batchSize,
                workerNumber,
                seqLen,
                seed
            );

            //Console.WriteLine($"等待");
            //Console.ReadLine();
            //Console.ReadLine();

            Console.WriteLine($"dataset initialized.词表大小:{dataset.VocabSize}");

            // 2. 初始化模型

            MiniLLM model;

            if (string.IsNullOrWhiteSpace(loadCurrentModel))
            {
                Console.WriteLine($"新建模型");

                model = new MiniLLM(dataset.VocabSize, contextLen, device, modelSize);    //新建
            }
            else
            {
                Console.WriteLine($"加载模型 {loadCurrentModel}");
                model = MiniLLM.LoadModel(loadCurrentModel, dataset.VocabSize, contextLen, device, modelSize);   //加载已有
            }

            MiniLLM modelGPU = model.to(device);
            await modelGPU.Train(dataset, numEpochs, learningRate, batchSize, verbose, checkPointSaveAfterEpochNum, checkPointSaveAfterEpochPeriod, promptForTestAfterEpochPeriodText, checkPointModel);


            // 6. 最终测试
            Console.WriteLine("训练完成");

            // 7. 保存模型
            // model.SaveModel(saveCurrentModel);
            // Console.WriteLine($"模型已保存到 {saveCurrentModel}");

            while (true)
            {
                string content = Console.ReadLine();

                if (content == "/q") break;
                string generated = modelGPU.GenerateTextFromPrompt(
                                                                    "<|im_start|>" + content,
                                                                    tokenizer,
                                                                     maxTokens: 100
                                                                 );
                Console.WriteLine(generated);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"程序运行出错: {ex.Message}\n{ex.StackTrace}");
        }
    }


    static async Task TrainLora(LLMConfig config)
    {


        // 配置
        Console.WriteLine(config.ToString());

        int contextLen = config.contextLen;    //128
        int batchSize = config.batchSize;
        int seqLen = config.seqLen;
        int numEpochs = config.numEpochs;
        float learningRate = config.learningRate;
        int verbose = config.verbose;
        int checkPointSaveAfterEpochNum = config.checkPointSaveAfterEpochNum;
        bool checkPointSaveAfterEpochPeriod = config.checkPointSaveAfterEpochPeriod;
        string promptForTestAfterEpochPeriodText = config.promptForTestAfterEpochPeriodText;
        string loadCurrentModel = config.loadCurrentModel;
        string checkPointModel = config.checkPointModel;
        string loraCheckPointModel = config.loraCheckPointModel;
        string saveCurrentModel = config.saveCurrentModel;
        int workerNumber = config.workerNumber ?? 3;
        ModelSize modelSize = (ModelSize)Enum.Parse(typeof(ModelSize), config.modelSize, ignoreCase: true);
        int loraRank = config.loraRank ?? 64;
        int loraAlpha = config.loraAlpha ?? 64;

        Console.WriteLine("配置加载完成");

        var data = JsonLReader_Lora.ConvertJsonlToStringArray(config.sftPath);

        try
        {

            // 1. 设置设备
            var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            Console.WriteLine($"Using device: {device}");
            var seed = 42;
            torch.random.manual_seed(seed);

            var tokenizer = new Tokenizer(config.tokenizerPath);
            Console.WriteLine($"Tokenizer initialized.");

            // 创建数据集（包含tokenizer）
            var dataset = new TextDataSet_Lora(
                device,
                tokenizer,
                data,
                batchSize,
                workerNumber,
                seqLen,
                seed
            );

            ////Console.WriteLine($"等待");
            ////Console.ReadLine();
            ////Console.ReadLine();

            Console.WriteLine($"dataset initialized.词表大小:{dataset.VocabSize}");

            // 2. 初始化模型

            MiniLLM model;

            
            Console.WriteLine($"加载模型 {loadCurrentModel}");
            model = MiniLLM.LoadModel(loadCurrentModel, dataset.VocabSize, contextLen, device, modelSize);   //加载已有

            if(model == null)
            {
                throw new Exception($"加载模型失败 {loadCurrentModel}");
            }

            LoraMiniLLM modelLora = new LoraMiniLLM(model,loraRank,loraAlpha);
            LoraMiniLLM modelLoraGPU = modelLora.to(device);
            await modelLoraGPU.Train(dataset, numEpochs, learningRate, batchSize, verbose, checkPointSaveAfterEpochNum, checkPointSaveAfterEpochPeriod, promptForTestAfterEpochPeriodText, checkPointModel, loraCheckPointModel);


            // 6. 最终测试
            Console.WriteLine("训练完成");

            // 7. 保存模型
            if (!string.IsNullOrWhiteSpace(saveCurrentModel))
            {
                model.SaveModel(saveCurrentModel);
                Console.WriteLine($"模型已保存到 {saveCurrentModel}");
            }

        }
        catch (Exception ex)
        {
            Console.WriteLine($"程序运行出错: {ex.Message}\n{ex.StackTrace}");
        }
    }


    static void Predict(LLMConfig config)
    {


        // 配置
        Console.WriteLine(config.ToString());

        int contextLen = config.contextLen;    //128
        int batchSize = config.batchSize;
        int seqLen = config.seqLen;
        int numEpochs = config.numEpochs;
        float learningRate = config.learningRate;
        int verbose = config.verbose;
        int checkPointSaveAfterEpochNum = config.checkPointSaveAfterEpochNum;
        bool checkPointSaveAfterEpochPeriod = config.checkPointSaveAfterEpochPeriod;
        string promptForTestAfterEpochPeriodText = config.promptForTestAfterEpochPeriodText;
        string loadCurrentModel = config.loadCurrentModel;
        string checkPointModel = config.checkPointModel;
        string saveCurrentModel = config.saveCurrentModel;
        int workerNumber = config.workerNumber ?? 3;
        ModelSize modelSize = (ModelSize)Enum.Parse(typeof(ModelSize), config.modelSize, ignoreCase: true);
        Console.WriteLine("配置加载完成");

        try
        {

            // 1. 设置设备
            var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            Console.WriteLine($"Using device: {device}");
            var seed = 42;
            torch.random.manual_seed(seed);

            // 创建数据集（包含tokenizer）
            var tokenizer = new Tokenizer(config.tokenizerPath);


            Console.WriteLine($"dataset initialized.词表大小:{tokenizer.GetVocabSize()}");

            // 2. 初始化模型

            MiniLLM model;

            Console.WriteLine($"加载模型 {loadCurrentModel}");
            model = MiniLLM.LoadModel(loadCurrentModel, tokenizer.GetVocabSize(), contextLen, device, modelSize);   //加载已有

            while (true)
            {
                string content = Console.ReadLine();

                if (content == "/q") break;
                string generated = model.GenerateTextFromPrompt(SftSample.Wrapper(content), tokenizer, maxTokens: 100, true);
                Console.WriteLine(generated);   // 结果：assistant 中国的首都是北京。);

            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"程序运行出错: {ex.Message}\n{ex.StackTrace}");
        }
    }
}
