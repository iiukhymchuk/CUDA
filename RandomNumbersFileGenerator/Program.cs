using System;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace RandomNumbersFileGenerator
{
    class Program
    {
        static void Main(string[] args)
        {
            var fileName = "file2_floats";
            var arrayLength = Math.Pow(2, 33); // 8589934592
            //var arrayLength = Math.Pow(2, 21);

            const int innerLoopLength = 8192;
            var outerLoopLength = arrayLength / innerLoopLength;

            var filePath1 = Environment.CurrentDirectory + @"\..\..\..\..\" + $@"\{fileName}.txt";

            var random = new Random();

            Stopwatch watch = new Stopwatch();

            watch.Start();

            using (StreamWriter outfile1 = new StreamWriter(filePath1))
            {
                var builder1 = new StringBuilder(innerLoopLength * 5);

                for (int i = 0; i < outerLoopLength; ++i)
                {
                    for (int j = 0; j < innerLoopLength; ++j)
                    {
                        var value = GetRandomInteger(random);
                        if (value < 0 || value >= 100)
                        {
                            Console.WriteLine(value);
                        }
                        builder1.AppendLine(value.ToString());
                    }

                    var text1 = builder1.ToString();

                    outfile1.Write(text1);

                    builder1.Length = 0;
                }
            }

            Console.WriteLine(watch.Elapsed.ToString());
            watch.Stop();
        }

        static int GetRandomInteger(Random random)
        {
            return random.Next(100);
        }
    }
}