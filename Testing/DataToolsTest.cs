using System;
using System.IO;
using DataUtils;

namespace Testing {

public class DataToolsTest {
	public static void run() {
		FileInfo inputFile = new ("data/heart.csv");
		FileInfo outputFile = new ("data/normal.csv");
		StrictDatasetNormalizer.fromStreamReaderToFile(inputFile.OpenText(), outputFile);
		Console.WriteLine("DONE");
	}
}

}