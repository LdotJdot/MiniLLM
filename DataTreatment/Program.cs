using LumTokenizer.Tokenizer;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;   // 如果 JSON 里还有结构，可再细化解析
using System.Text.RegularExpressions;

namespace DataTreatment
{
    class Program
    {
        // 输入、输出路径
        private const string InputPath = @"D:\Data\Personal\AI\llm\pretrain_hq.jsonl";
        private const string PureTextDataPath = @"D:\Data\Personal\AI\llm\pretrain_hq_extracted.txt";
        private const string OutputPathEncode = @"D:\Data\Personal\AI\llm\pretrain_hq_encoded.txt";

        static void Main()
        {
            Trans2PureText(InputPath);
            Trans2Encode(PureTextDataPath);
        }

        static void Trans2Encode(string path)
        {
            if (!File.Exists(path))
            {
                Console.WriteLine("源文件不存在：" + path);
                return;
            }

            // 如果输出目录不存在，先创建
            Directory.CreateDirectory(Path.GetDirectoryName(OutputPathEncode)!);

            // 用 StreamWriter 写新文件；UTF-8 无 BOM
            using var writer = new StreamWriter(OutputPathEncode, append: false, Encoding.UTF8);

            long totalBytes = new FileInfo(path).Length;
            long processedBytes = 0;

            int lineCount = 0;

            var tokenizer = BPETokenizer.CreateTokenizer("D:\\Data\\Personal\\AI\\llm\\tokenizer\\minimind_tokenizer.txt");

                var strarr= File.ReadLines(path).ToArray();
            for ( int i=0;i<strarr.Length;i++ )
            {
                var line=strarr[i];

                if (string.IsNullOrWhiteSpace(line)) continue;

                var res = tokenizer.Encode(line, false);

                writer.WriteLine($"[{string.Join(',', res)}]");
                // 累加字节数（+2 是 \r\n 长度，粗略即可）
                processedBytes += Encoding.UTF8.GetByteCount(line) + 2;
                lineCount++;

                // 每 10 000 行刷一次进度，避免刷屏
                if (lineCount % 1_000 == 0)
                {
                    double percent = 100.0 * processedBytes / totalBytes;
                    Console.Write($"\rProgress: {percent,5:F1}%  ({lineCount:N0} lines), line:{i}");
                }
            }
            // 最后把光标移到下一行
            Console.WriteLine();
        }
        static void Trans2PureText(string path)
        {
            if (!File.Exists(path))
            {
                Console.WriteLine("源文件不存在：" + path);
                return;
            }

            // 如果输出目录不存在，先创建
            Directory.CreateDirectory(Path.GetDirectoryName(PureTextDataPath)!);

            // 用 StreamWriter 写新文件；UTF-8 无 BOM
            using var writer = new StreamWriter(PureTextDataPath, append: false, Encoding.UTF8);

            long totalBytes = new FileInfo(path).Length;
            long processedBytes = 0;

            int lineCount = 0;
            foreach (var line in File.ReadLines(path))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;

                var blocks = Convert(line);
                foreach (var b in blocks)
                {
                    writer.WriteLine(b);
                }

                // 累加字节数（+2 是 \r\n 长度，粗略即可）
                processedBytes += Encoding.UTF8.GetByteCount(line) + 2;
                lineCount++;

                // 每 10 000 行刷一次进度，避免刷屏
                if (lineCount % 10_000 == 0)
                {
                    double percent = 100.0 * processedBytes / totalBytes;
                    Console.Write($"\rProgress: {percent,5:F1}%  ({lineCount:N0} lines)");
                }
            }
            // 最后把光标移到下一行
            Console.WriteLine();
        }

        /// <summary>
        /// 非贪婪提取 <|im_start|>…<|im_end|> 之间的内容
        /// </summary>
        static List<string> Convert(string text)
        {
            var matches = Regex.Matches(
                text,
                @"<\|im_start\|>(.*?)<\|im_end\|>",
                RegexOptions.Singleline);

            var blocks = new List<string>(matches.Count);
            foreach (Match m in matches)
                blocks.Add(m.Groups[1].Value.Trim());

            return blocks;
        }
    }
}