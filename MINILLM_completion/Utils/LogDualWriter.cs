using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;

namespace MINILLM_Completion.Utils
{
    /// <summary>
    /// 把控制台输出同时镜像到日志文件（按天滚动，进程级唯一）
    /// 用法：LogDualWriter.Attach();   // 程序入口调用一次即可
    /// </summary>
    public sealed class LogDualWriter : TextWriter
    {
        private static readonly Lazy<LogDualWriter> _instance =
            new Lazy<LogDualWriter>(() => new LogDualWriter(), LazyThreadSafetyMode.ExecutionAndPublication);

        private readonly TextWriter _consoleOut;          // 原控制台
        private StreamWriter _fileWriter;                 // 文件流
        private readonly Lock _lock = new ();

        private LogDualWriter()
        {
            _consoleOut = Console.Out;
            RollFile();                                   // 创建今天文件
            Console.SetOut(this);                         // 替换全局 Out
        }

        public static void Attach(string? folder = null)
        {
            _ = _instance.Value; // 触发单例
        }

        // 按程序启动时间（秒级）命名，只滚一次
        private void RollFile()
        {
            if (_fileWriter != null) return;   // 已经初始化过就不再滚

            var folder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "logs");
            Directory.CreateDirectory(folder);

            // 文件名 = 启动时间精确到秒
            var fileName = Path.Combine(folder,
                $"{Process.GetCurrentProcess().StartTime:yyyyMMdd_HHmmss}.log");

            lock (_lock)
            {
                if (_fileWriter != null) return;
                _fileWriter = new StreamWriter(
                                                new FileStream(fileName,
                                                FileMode.Append,
                                                FileAccess.Write,
                                                FileShare.Read),       // 关键点：允许其它进程同时只读
                                                Encoding.UTF8);            // 无 BOM，方便任意编辑器查看
                _fileWriter.AutoFlush = true;
            }
        }

        // 核心写方法
        public override void Write(char value)
        {
            _consoleOut.Write(value);                     // 先写控制台
            lock (_lock)
            {
                RollFile();                               // 跨天时切换文件
                _fileWriter.Write(value);
            }
        }

        public override void Write(string? value)
        {
            _consoleOut.Write(value);
            lock (_lock)
            {
                RollFile();
                _fileWriter.Write(value);
            }
        }

        public override void WriteLine(string? value)
        {
            _consoleOut.WriteLine(value);
            lock (_lock)
            {
                RollFile();
                _fileWriter.WriteLine($"[{DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff")}] {value}");
            }
        }

        public override Encoding Encoding => Encoding.UTF8;

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                lock (_lock)
                {
                    _fileWriter?.Dispose();
                }
            }
            base.Dispose(disposing);
        }
    }
}