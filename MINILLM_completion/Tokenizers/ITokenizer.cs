using System;
using System.Collections.Generic;
using System.Text;

namespace MINILLM_Completion.Tokenizers
{
    public interface ITokenizer
    {
 

        public List<int> Encode(string text);
        public string Decode(IList<int> tokens, bool includeSpecial = true);

        public int GetToken(string token);
        public bool IsEndToken(int token);
        public int GetVocabSize();

    }
}
