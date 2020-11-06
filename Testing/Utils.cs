using System.Collections.Generic;
using System.Drawing;
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

	public static void writeToExcel<T>(NNsData<T> data, string rootPath) {
		createRootResultsDirectoryIfNotExists(rootPath);

		List<int> correctAnswersAmount = data.correctAnswersAmount;
		List<T> correctAnswers = data.correctAnswers;
		List<List<T>> answers = data.answers;

		ExcelPackage.LicenseContext = LicenseContext.NonCommercial;
		using ExcelPackage package = new ExcelPackage(new FileInfo($"{rootPath}/test_results.xlsx"));
		ExcelWorksheet worksheet = package.Workbook.Worksheets.Add("Results");

		worksheet.Cells[1, 1].Value = "EXP";
		worksheet.Cells[1, 1].Style.Font.Bold = true;
		worksheet.Cells[2, 1].Value = correctAnswers.Count;
		worksheet.Column(1).Style.Border.Right.Style = ExcelBorderStyle.Thin;

		for (int i = 0; i < correctAnswers.Count; i++)
			worksheet.Cells[i + 4, 1].Value = correctAnswers[i];

		for (int nnIndex = 0; nnIndex < answers.Count; nnIndex++) {
			int currentColumn = nnIndex + 2;

			worksheet.Cells[1, currentColumn].Value = "NN " + (nnIndex + 1);
			worksheet.Cells[1, currentColumn].Style.Font.Bold = true;

			worksheet.Cells[2, currentColumn].Value = correctAnswersAmount[nnIndex];
			worksheet.Cells[3, currentColumn].Value = correctAnswersAmount[nnIndex] * 1d / answers[nnIndex].Count;

			for (int answerIndex = 0; answerIndex < answers[nnIndex].Count; answerIndex++) {
				worksheet.Cells[4 + answerIndex, currentColumn].Value = answers[nnIndex][answerIndex];
				if (!answers[nnIndex][answerIndex].Equals(correctAnswers[answerIndex])) 
					worksheet.Cells[4 + answerIndex, currentColumn].Style.Font.Color.SetColor(Color.Red);
			}

			worksheet.Column(currentColumn).Style.Border.Right.Style = ExcelBorderStyle.Thin;
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

public class NNsData <T> {
	public List<int> correctAnswersAmount { get; }
	public List<T> correctAnswers { get; }
	public List<List<T>> answers { get; }

	public NNsData(List<int> correctAnswersAmount, List<T> correctAnswers, List<List<T>> answers) {
		this.correctAnswersAmount = correctAnswersAmount;
		this.correctAnswers = correctAnswers;
		this.answers = answers;
	}
}

}