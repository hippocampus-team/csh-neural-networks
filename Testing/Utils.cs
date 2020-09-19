using System.Collections.Generic;
using System.IO;
using NeuralNetworks;
using OfficeOpenXml;
using OfficeOpenXml.Style;

namespace Testing {

public class Utils {
	public static int getNextByte(Stream fileStream) {
		byte[] digit = new byte[1];
		fileStream.Read(digit, 0, 1);
		return digit[0];
	}

	public static void writeToExcel(NNsData data, string rootPath) {
		createRootResultsDirectoryIfNotExists(rootPath);

		List<int> numOfGood = data.numOfGood;
		List<List<KeyValuePair<int, int>>> memory = data.memory;

		ExcelPackage.LicenseContext = LicenseContext.NonCommercial;
		using ExcelPackage package = new ExcelPackage(new FileInfo($"{rootPath}/test_results.xlsx"));
		ExcelWorksheet worksheet = package.Workbook.Worksheets.Add("Results");

		for (int i = 0; i < memory.Count; i++) {
			int currentColumn = i * 2 + 1;

			worksheet.Cells[1, currentColumn].Value = "ANS" + (i + 1);
			worksheet.Cells[1, currentColumn].Style.Font.Bold = true;
			worksheet.Cells[1, currentColumn + 1].Value = "EXP" + (i + 1);
			worksheet.Cells[1, currentColumn + 1].Style.Font.Bold = true;

			worksheet.Cells[2, currentColumn].Value = numOfGood[i];
			worksheet.Cells[2, currentColumn + 1].Value = memory[i].Count;
			worksheet.Cells[3, currentColumn + 1].Value = numOfGood[i] * 1d / memory[i].Count;

			for (int index = 0; index < memory[i].Count; index++) {
				(int key, int value) = memory[i][index];

				worksheet.Cells[4 + index, currentColumn].Value = key;
				worksheet.Cells[4 + index, currentColumn + 1].Value = value;
			}

			worksheet.Column(currentColumn + 1).Style.Border.Right.Style = ExcelBorderStyle.Thin;
		}

		package.Save();
	}

	public static void writeNNsToFiles(List<NeuralNetwork> nns, string rootPath) {
		createRootResultsDirectoryIfNotExists(rootPath);

		for (int i = 0; i < nns.Count; i++)
			File.WriteAllText($"{rootPath}/nn_{i}.json", nns[i].serialize());
	}

	private static void createRootResultsDirectoryIfNotExists(string rootPath) {
		if (!Directory.Exists(rootPath)) Directory.CreateDirectory(rootPath);
	}
}

}