using LumTokenizer.Tokenizer;
using System;
using System.Collections.Generic;
using System.Text;
using static System.Net.Mime.MediaTypeNames;

namespace MINILLM_Completion.Tokenizers
{
    public class Tokenizer : ITokenizer
    {
        ConcurrentBPETokenizer tokenizer;

        public const string imstart = "<|im_start|>";
        public const string imend = "<|im_end|>";
        public const string eot = "<|endoftext|>";
        public const int token_imstart = 1;
        public const int token_imend = 2;
        public const int token_eot = 0;


        public Tokenizer(string tokenizerFilePath)
        {
            this.tokenizer = ConcurrentBPETokenizer.CreateTokenizer(tokenizerFilePath);
        }

        public List<int> Encode(string text)
        {
            var list = tokenizer.Encode(text, true);
            return list;
        }

        public string Decode(IList<int> tokens, bool includeSpecial = true)
        {
            return tokenizer.Decode(tokens, includeSpecial);
        }

        public int GetToken(string token)
        {
            throw new NotImplementedException();
        }

        public bool IsEndToken(int token)
        {
            return token == token_eot || token == token_imend;
        }

        public int GetVocabSize()
        {
            return tokenizer.VocabSize;
        }
    }
}
