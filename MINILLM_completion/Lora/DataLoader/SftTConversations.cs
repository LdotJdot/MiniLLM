using MINILLM_Completion.Tokenizers;
using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json.Serialization;

namespace MINILLM_Completion.Lora.DataLoader
{
    /// <summary>
    /// 单条 SFT 样本
    /// </summary>
    public class SftSample
    {
        [JsonPropertyName("conversations")]
        public List<SftTurn> Conversations { get; set; } = new();


        public const string END_OF_TEXT = Tokenizer.eot;
        public const string IMSTART = Tokenizer.imstart;
        public const string IMEND = Tokenizer.imend;
        public const string USER_TOKEN = "user";
        public const string ASSISTANT_TOKEN = "assistant";

        StringBuilder stringCache = new StringBuilder();

        public static string Wrapper(string? content)
        {
            return $"{SftSample.IMSTART}{SftSample.USER_TOKEN} {content ?? string.Empty}{SftSample.IMEND}";
        }

        public override string ToString()
        {
            stringCache.Clear();
            for (int i = 0; i < Conversations.Count; i++)
            {
                var t = Conversations[i];
                if (t.Role == USER_TOKEN)
                {
                    stringCache.Append(IMSTART).Append(USER_TOKEN).Append(' ').Append(t.Content.Trim()).Append(IMEND);
                }
                else if (t.Role == ASSISTANT_TOKEN || t.Role == "bot")
                {
                    stringCache.Append(Tokenizer.imstart).Append(ASSISTANT_TOKEN).Append(' ').Append(t.Content.Trim()).Append(IMEND);
                    // 只有 assistant 回复后面才加 EOS/EOT，方便语言模型知道「说完」
                    stringCache.Append(END_OF_TEXT);
                }
                // system 角色或其他角色可自行扩展
            }

            return stringCache.ToString();
        }
    }

    public class SftTurn
    {
        [JsonPropertyName("role")] public string Role { get; set; } = "";
        [JsonPropertyName("content")] public string Content { get; set; } = "";
    }
}
