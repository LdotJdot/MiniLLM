using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;

namespace MINILLM_Completion.Utils.DataLoader
{
    public sealed class EpochCircularSampler<T>
    {
        private readonly T[] _data;
        private readonly int _len;
        private long _cursor;      // 下一次要采的起始序号（元素单位）
        private int _epoch;        // 已完成洗牌轮数（诊断用）
        private volatile int _shuffling; // 0-空闲 1-正在洗牌

        public EpochCircularSampler(IEnumerable<T> source)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            _data = source.ToArray();
            _len = _data.Length;
            if (_len == 0) throw new InvalidOperationException("Source array is empty.");
            Shuffle(_data);
        }

        /// <summary>
        /// 取一个 batch。如果本次调用正好完成一整轮，会在返回前自动洗牌并重置游标。
        /// </summary>
        public T[] TakeBatch(int batchSize)
        {
            if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));

            // 1. 原子“批扣”batchSize 个位置
            long last = Interlocked.Add(ref _cursor, batchSize) - 1;
            long first = last - batchSize + 1;

            // 2. 计算是否刚采完一整圈
            long completedRounds = last / _len;
            long previousRounds = (first - 1) / _len;
            bool finishedOneEpoch = completedRounds != previousRounds;

            // 3. 采数据
            T[] batch = new T[batchSize];
            for (int i = 0; i < batchSize; i++)
                batch[i] = _data[FloorMod(first + i, _len)];

            // 4. 如果刚采完一圈，尝试洗牌 + 原子回卷游标（仅一个线程成功）
            if (finishedOneEpoch)
            {
                if (Interlocked.CompareExchange(ref _shuffling, 1, 0) == 0)
                {
                    try
                    {
                        Shuffle(_data);
                        // 关键修正：把游标回卷到当前圈的偏移量，而非剩余量
                        long newCursor = FloorMod(last + 1, _len);
                        // CAS 循环，确保不会跳过任何元素
                        long snapshot;
                        do
                        {
                            snapshot = Volatile.Read(ref _cursor);
                        } while (snapshot != newCursor &&
                                 Interlocked.CompareExchange(ref _cursor, newCursor, snapshot) != snapshot);

                        Interlocked.Increment(ref _epoch);
                    }
                    finally
                    {
                        Volatile.Write(ref _shuffling, 0);
                    }
                }
            }

            return batch;
        }

        /// <summary>当前已完成的洗牌轮数（外部可读）</summary>
        public int Epoch => _epoch;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int FloorMod(long x, int mod)
        {
            long rem = x % mod;
            if (rem < 0) rem += mod;
            return (int)rem;
        }

        private static void Shuffle(T[] array)
        {
            for (int i = array.Length - 1; i > 0; i--)
            {
                int j = Random.Shared.Next(i + 1);
                (array[i], array[j]) = (array[j], array[i]);
            }
        }
    }
}
