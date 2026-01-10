// dotnet run --input yourfile.jsonl
using System;
using System.IO;
using System.Text.RegularExpressions;
using System.Collections.Generic;
using System.Text.Json;
using MINILLM_Completion.Lora.DataLoader;

public record struct textData
{
    public string text { get; set; }
}

public class JsonLReader_Pretrain
{
    /// <summary>
    /// 把 jsonl 转成 string[]，每个元素是一次完整多轮对话的纯文本
    /// </summary>
    public static List<string> ConvertJsonlToStringArray(string jsonlPath)
    {
        var results = new List<string>();
        foreach (string line in File.ReadLines(jsonlPath))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;

            var text = JsonSerializer.Deserialize<textData>(line).text;

            if(!string.IsNullOrWhiteSpace(text)) results.Add(text);
        }
        return results;
    }

    /// <summary>
    /// 把 jsonl 转成 string[]，每个元素是一次完整多轮对话的纯文本
    /// </summary>
    public static List<List<int>> ConvertJsonlToIntArrayArray(string jsonlPath)
    {
        var results = new List<List<int>>();
        foreach (string line in File.ReadLines(jsonlPath))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;

            var value = JsonSerializer.Deserialize<List<int>>(line);

            if (value!=null && value.Count>0) results.Add(value);
        }
        return results;
    }
}