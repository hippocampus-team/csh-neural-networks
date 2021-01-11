using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace DataUtils {
    public static class StrictDatasetNormalizer {
        public static void fromStreamReaderToFile(StreamReader data, FileInfo outputFile) {
            StreamWriter output = new (outputFile.FullName);
            Dictionary<int, Stack<string>> sortedData = new ();
            
            // Copy header line
            output.WriteLine(data.ReadLine());

            while (!data.EndOfStream) {
                string entery = data.ReadLine();
                if (entery == null) break;
                int label = int.Parse(entery.Substring(entery.LastIndexOf(',') + 1));

                if (sortedData.ContainsKey(label)) sortedData[label].Push(entery);
                else sortedData.Add(label, new Stack<string>(new [] { entery }));
            }

            int minLabelAmount = sortedData.Min(pair => pair.Value.Count);

            for (int i = 0; i < minLabelAmount; i++)
                foreach (Stack<string> labelEntries in sortedData.Values)
                    output.WriteLine(labelEntries.Pop());

            output.Close();
        }
    }
}