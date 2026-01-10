using MemoryPack;
using MINILLM_Completion.Utils;
using System.Text.Json;
using static LLM.ModelSizeTable;

namespace MINILLM_Completion.LLM.Config
{
    public partial class LLMConfig
    {
        // 配置
        public int contextLen { get; set; } = 64;    //128
        public int batchSize { get; set; } = 128;
        public int seqLen { get; set; } = 64;
        public int numEpochs { get; set; } = 1;
        public float learningRate { get; set; } = 1E-1f;
        public int verbose { get; set; } = 20;
        public int checkPointSaveAfterEpochNum { get; set; } = 1;
        public bool checkPointSaveAfterEpochPeriod { get; set; } = true;
        public string pretrainPath { get; set; } = string.Empty;
        public string? sftPath { get; set; } = string.Empty;
        public string tokenizerPath { get; set; } = string.Empty;
        public string promptForTestAfterEpochPeriodText { get; set; } = string.Empty;
        public string loadCurrentModel { get; set; } = string.Empty;
        public string checkPointModel { get; set; } = string.Empty;
        public string loraCheckPointModel { get; set; } = string.Empty;
        public string saveCurrentModel { get; set; } = string.Empty;
        public string modelSize { get; set; } = "n";
        public int? workerNumber { get; set; } = 6;
        public int? loraRank { get; set; }
        public int? loraAlpha { get; set; }

        public static LLMConfig? Load(string path)
        {
            return JsonSerializer.Deserialize<LLMConfig>(File.ReadAllBytes(path));
        }

        public void Save(string path)
        {
            File.WriteAllText(path, JsonSerializer.Serialize<LLMConfig>(this));
        }

        public override string ToString()
        {
            var sb = new System.Text.StringBuilder();
            sb.AppendLine("LLMConfig:");
            sb.AppendLine($"  contextLen                    : {contextLen}");
            sb.AppendLine($"  batchSize                     : {batchSize}");
            sb.AppendLine($"  seqLen                        : {seqLen}");
            sb.AppendLine($"  numEpochs                     : {numEpochs}");
            sb.AppendLine($"  learningRate                  : {learningRate}");
            sb.AppendLine($"  verbose                       : {verbose}");
            sb.AppendLine($"  checkPointSaveAfterEpochNum   : {checkPointSaveAfterEpochNum}");
            sb.AppendLine($"  checkPointSaveAfterEpochPeriod: {checkPointSaveAfterEpochPeriod}");
            sb.AppendLine($"  pretrainPath                  : {pretrainPath ?? "(null)"}");
            sb.AppendLine($"  tokenizerPath                 : {tokenizerPath ?? "(null)"}");
            sb.AppendLine($"  promptForTestAfterEpochPeriodText: {promptForTestAfterEpochPeriodText ?? "(null)"}");
            sb.AppendLine($"  loadCurrentModel              : {loadCurrentModel ?? "(null)"}");
            sb.AppendLine($"  checkPointModel               : {checkPointModel ?? "(null)"}");
            sb.AppendLine($"  saveCurrentModel              : {saveCurrentModel ?? "(null)"}");
            sb.AppendLine($"  modelSize                     : {modelSize ?? "(null)"}");
            sb.AppendLine($"  workerNumber                  : {workerNumber?.ToString() ?? "(null)"}");
            sb.AppendLine($"  loraRank                      : {loraRank?.ToString() ?? "(null)"}");
            sb.AppendLine($"  loraAlpha                     : {loraAlpha?.ToString() ?? "(null)"}");
;
            return sb.ToString();
        }
    }
}