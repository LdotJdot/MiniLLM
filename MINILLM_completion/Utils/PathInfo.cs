using System;
using System.Collections.Generic;
using System.Text;

namespace MINILLM_Completion.Utils
{
    public class PathHelper
    {
        public static string BasePath = AppDomain.CurrentDomain.BaseDirectory;
        public static string TokenPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory,"tokens.dat");
        public static string ConfigPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory,"lumConf.Json");
    }
}
