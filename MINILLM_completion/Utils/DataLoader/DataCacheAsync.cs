
using System.Threading.Channels;

public class DataCacheAsync<T>: IAsyncDisposable
{

    Channel<T> dataCache;

    int capacity;

    public int Capacity => capacity;
    CancellationTokenSource cts;
    private Task[] producerTasks;

    public DataCacheAsync(int capacity)
    {
        this.capacity = capacity;

        dataCache = Channel.CreateBounded<T>(new BoundedChannelOptions(capacity)
        {
            FullMode = BoundedChannelFullMode.Wait   // 写端阻塞
        });

        cts = new CancellationTokenSource();
    }

    public void StartProducer(Func<T> writeAction, int workerNumber)
    {
        producerTasks= new Task[workerNumber];
        for (int i = 0; i < workerNumber; i++)
        {
            producerTasks[i] = Task.Run(() => Producer(writeAction, cts.Token));
        }
    }

    long i = 0;
    private async Task Producer(Func<T> writeAction, CancellationToken token)
    {
        var writer = dataCache.Writer;
        try
        {
            while (!token.IsCancellationRequested)
            {
                await writer.WriteAsync(writeAction(), token).ConfigureAwait(false);      // 容量满时会在此处挂起
                //Console.WriteLine($"Produced {++i} items.");
            }
        }
        catch(Exception ex)
        {
            Console.WriteLine(ex.Message);
        }
        finally
        {
            writer.TryComplete();
        }
    }

    public async Task<T> Consume()
    {
        var item = await dataCache.Reader.ReadAsync().ConfigureAwait(false);
        return item;
    }
 
    public async ValueTask DisposeAsync()
    {
        // 1. 通知停工
        cts.Cancel();

        // 2. 等生产/消费任务自己退完
        await Task.WhenAll(producerTasks)
                  .ConfigureAwait(false);

        // 3. 释放 CTS
        cts.Dispose();
    }
}




