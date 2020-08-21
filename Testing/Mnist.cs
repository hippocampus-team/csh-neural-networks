using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NeuralNetworks;
using NeuralNetworks.Misc;
using NeuralNetworks.Units;
using OfficeOpenXml;
using OfficeOpenXml.Style;

namespace Testing {

internal static class Mnist {
	private static string rootPath;
	private static string experimentTitle;
	private static List<NeuralNetwork> nns;

	public static void run() {
		Console.Write("Hello. Enter experiment title: ");
		experimentTitle = Console.ReadLine();
		rootPath = $"./results/{experimentTitle}";

		Console.Write("Initialisation of NNs...");
		setupNNs();
		Console.WriteLine(" Done.");

		Console.WriteLine("Training started!");
		Console.Write("In progress");
		train(8);
		Console.WriteLine(" Done.");

		Console.WriteLine("Testing started!");
		Console.Write("In progress");
		NNsData testData = test(2);
		Console.WriteLine(" Done.");

		Console.Write("Writing data to excel... ");
		writeToExcel(testData);
		Console.WriteLine(" Done.");

		Console.Write("Writing NNs configurations to files... ");
		writeNNsToFiles();
		Console.WriteLine(" Done.");

		Console.WriteLine("Process is completed and result is saved in excel file.");
		Console.ReadKey();
	}

	private static void setupNNs() {
		nns = new List<NeuralNetwork>();
		for (int i = 0; i < 6; i++) {
			NeuralNetwork nn = new NeuralNetwork();

			nn.setInputLength(784);
			nn.addDenceLayer(300, new ModifiedSigmoid());
			nn.addDenceLayer(10, new ModifiedSigmoid());
			
			///// LeNet-5 ? 
			// nn.setInputLength(784);
			// nn.addPoolingLayer(new Filter(2), 2, PoolingNode.PoolingMethod.average);
			// nn.addConvolutionalLayer(new Filter(5), 16, 1);
			// nn.addPoolingLayer(new Filter(2), 2, PoolingNode.PoolingMethod.average);
			// nn.addDenceLayer(120);
			// nn.addDenceLayer(84);
			// nn.addDenceLayer(10);

			nn.fillRandomWeights();
			nn.fillRandomBiases();

			nns.Add(nn);
		}
	}

	private static void train(int iterations) {
		FileStream trainImages = new FileStream("data/train_imgs", FileMode.Open);
		FileStream trainLabels = new FileStream("data/train_lbls", FileMode.Open);
		trainImages.Read(new byte[4 * 4], 0, 4 * 4);
		trainLabels.Read(new byte[4 * 2], 0, 4 * 2);

		EList<double> answer = new EList<double>();
		for (int i = 0; i < 10; i++) answer.Add(0.5);

		EList<double> input = new EList<double>();
		for (int i = 0; i < 784; i++) input.Add(0);

		for (int h = 0; h < iterations; h++) {
			for (int i = 0; i < 1000; i++) {
				int digit = getNextByte(trainLabels);

				byte[] byteInput = new byte[784];
				trainImages.Read(byteInput, 0, 784);
				for (int j = 0; j < 784; j++) input[j] = byteInput[j] / 255d;

				foreach (NeuralNetwork nn in nns) {
					nn.putData(input);
					nn.run();
				}

				answer[digit] = 1;
				foreach (NeuralNetwork nn in nns) nn.backpropagate(answer, 1);
				answer[digit] = 0.4;
			}
			Console.Write(".");
		}
	}

	private static NNsData test(int iterations) {
		FileStream testImages = new FileStream("data/test_imgs", FileMode.Open);
		FileStream testLabels = new FileStream("data/test_lbls", FileMode.Open);
		testImages.Read(new byte[4 * 4], 0, 4 * 4);
		testLabels.Read(new byte[4 * 2], 0, 4 * 2);

		EList<double> input = new EList<double>();
		for (int i = 0; i < 784; i++) input.Add(0);

		List<int> numOfGood = nns.Select(nn => 0).ToList();
		List<List<KeyValuePair<int, int>>> memory = nns.Select(nn => new List<KeyValuePair<int, int>>()).ToList();

		for (int h = 0; h < iterations; h++) {
			for (int i = 0; i < 1000; i++) {
				int digit = getNextByte(testLabels);

				byte[] byteInput = new byte[784];
				testImages.Read(byteInput, 0, 784);
				for (int j = 0; j < 784; j++) input[j] = byteInput[j] / 255d;

				for (int nnIndex = 0; nnIndex < nns.Count; nnIndex++) {
					NeuralNetwork nn = nns[nnIndex];

					nn.putData(input);
					nn.run();

					int answer = nn.getMaxIndexInOutput();
					if (answer == digit) numOfGood[nnIndex]++;

					memory[nnIndex].Add(new KeyValuePair<int, int>(answer, digit));
				}
			}
			Console.Write(".");
		}

		return new NNsData(numOfGood, memory);
	}

	private static int getNextByte(Stream fileStream) {
		byte[] digit = new byte[1];
		fileStream.Read(digit, 0, 1);
		return digit[0];
	}

	private static void writeToExcel(NNsData data) {
		createRootResultsDirectoryIfNotExists();

		List<int> numOfGood = data.numOfGood;
		List<List<KeyValuePair<int, int>>> memory = data.memory;

		ExcelPackage.LicenseContext = LicenseContext.NonCommercial;
		using ExcelPackage package = new ExcelPackage(new FileInfo($"{rootPath}/test_results.xlsx"));
		ExcelWorksheet worksheet = package.Workbook.Worksheets.Add(experimentTitle);

		for (int i = 0; i < nns.Count; i++) {
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

	private static void writeNNsToFiles() {
		createRootResultsDirectoryIfNotExists();

		for (int i = 0; i < nns.Count; i++)
			File.WriteAllText($"{rootPath}/nn_{i}.json", nns[i].serialize());
	}

	private static void createRootResultsDirectoryIfNotExists() {
		if (!Directory.Exists(rootPath)) Directory.CreateDirectory(rootPath);
	}
}

internal class NNsData {
	public List<int> numOfGood { get; }
	public List<List<KeyValuePair<int, int>>> memory { get; }

	public NNsData(List<int> numOfGood, List<List<KeyValuePair<int, int>>> memory) {
		this.numOfGood = numOfGood;
		this.memory = memory;
	}
}

}