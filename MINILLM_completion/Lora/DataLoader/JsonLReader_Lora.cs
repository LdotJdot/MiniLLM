// dotnet run --input yourfile.jsonl
using System;
using System.IO;
using System.Text.RegularExpressions;
using System.Collections.Generic;
using System.Text.Json;
using MINILLM_Completion.Lora.DataLoader;


public class JsonLReader_Lora
{
    // <summary>
    /// 把 jsonl 转成 string[]，每个元素是一次完整多轮对话的纯文本
    /// </summary>
    public static List<string> ConvertJsonlToStringArray(string jsonlPath)
    {
        var results = new List<string>();
        foreach (string line in File.ReadLines(jsonlPath))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;

            var sample = JsonSerializer.Deserialize<SftSample>(line);

            if (sample!=null) results.Add(sample.ToString());
        }
        return results;
    }


}