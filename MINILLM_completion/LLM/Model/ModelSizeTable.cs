using System;
using System.Collections.Generic;
using System.Text;

namespace LLM
{
    // <summary>
    /// 网络规模枚举
    /// </summary>
    public enum ModelSize:int
    {
        n,   // nano
        s,   // small
        m,   // medium
        l,   // large
        x    // x-large
    }

    /// <summary>
    /// 把枚举映射到超参
    /// </summary>
    internal static class ModelSizeTable
    {
        public struct Config
        {
            public int n_layer;
            public int n_head;
            public int n_embd;
            public int ffn_hidden => 4 * n_embd;
        }

        private static readonly Dictionary<ModelSize, Config> _dict =
            new Dictionary<ModelSize, Config>
            {
                [ModelSize.n] = new Config { n_layer = 6, n_head = 8, n_embd = 256 },
                [ModelSize.s] = new Config { n_layer = 8, n_head = 8, n_embd = 512 },
                [ModelSize.m] = new Config { n_layer = 12, n_head = 12, n_embd = 768 },
                [ModelSize.l] = new Config { n_layer = 24, n_head = 16, n_embd = 1024 },
                [ModelSize.x] = new Config { n_layer = 36, n_head = 20, n_embd = 1280 }
            };

        public static Config Lookup(ModelSize size) => _dict[size];
    }
}
