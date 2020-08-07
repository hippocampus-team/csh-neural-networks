using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NeuralNetworks;
using NeuralNetworks.Misc;
using OfficeOpenXml;
using OfficeOpenXml.Style;

namespace Testing {
	internal static class Mnist {
		private static string experimentTitle;
		private static List<NeuralNetwork> nns;
		
		public static void Run() {
			Console.Write("Hello. Enter experiment title: ");
			experimentTitle = Console.ReadLine();
			
			Console.Write("Initialisation of NNs...");
			SetupNNs();
			Console.WriteLine(" Done.");
			
			Console.WriteLine("Training started!");
			Console.Write("In progress");
			Train();
			Console.WriteLine(" Done.");

			Console.WriteLine("Testing started!");
			Console.Write("In progress");
			Test();
			Console.WriteLine(" Done.");
			
			Console.WriteLine("Process is completed and result is saved in excel file.");
			Console.ReadKey();
		}

		private static void SetupNNs() {
			nns = new List<NeuralNetwork>();
			for (int i = 0; i < 4; i++) {
				NeuralNetwork nn = new NeuralNetwork(true);

				nn.SetInputLength(784);
				nn.AddDenceLayer(300);
				nn.AddDenceLayer(10);

				nn.FillRandomWeights();
				nn.FillRandomBiases();
				
				nns.Add(nn);
			}
		}

		private static void Train() {
			FileStream trainImages = new FileStream("data/train_imgs", FileMode.Open);
			FileStream trainLabels = new FileStream("data/train_lbls", FileMode.Open);
			trainImages.Read(new byte[4 * 4], 0, 4 * 4);
			trainLabels.Read(new byte[4 * 2], 0, 4 * 2);
			
			EList<double> answer = new EList<double>();
			for (int i = 0; i < 10; i++) answer.Add(0.5);
			
			EList<double> input = new EList<double>();
			for (int i = 0; i < 784; i++) input.Add(0);
			
			for (int h = 0; h < 15; h++) {
				for (int i = 0; i < 1000; i++) {
					int digit = GetNextByte(trainLabels);
				
					byte[] byteInput = new byte[784];
					trainImages.Read(byteInput, 0, 784);
					for (int j = 0; j < 784; j++) input[j] = byteInput[j] / 255d;

					foreach (NeuralNetwork nn in nns) {
						nn.PutData(input);
						nn.Run();
					}

					answer[digit] = 1;
					foreach (NeuralNetwork nn in nns) nn.Backpropagate(answer, 1);
					answer[digit] = 0.5;
				}
				
				Console.Write(".");
			}
		}

		private static void Test() {
			FileStream testImages = new FileStream("data/test_imgs", FileMode.Open);
			FileStream testLabels = new FileStream("data/test_lbls", FileMode.Open);
			testImages.Read(new byte[4 * 4], 0, 4 * 4);
			testLabels.Read(new byte[4 * 2], 0, 4 * 2);

			EList<double> input = new EList<double>();
			for (int i = 0; i < 784; i++) input.Add(0);
			
			List<int> numOfGood = nns.Select(nn => 0).ToList();
			List<List<KeyValuePair<int, int>>> memory = 
				nns.Select(nn => new List<KeyValuePair<int, int>>()).ToList();

			for (int h = 0; h < 5; h++) {
				for (int i = 0; i < 1000; i++) {
					int digit = GetNextByte(testLabels);

					byte[] byteInput = new byte[784];
					testImages.Read(byteInput, 0, 784);
					for (int j = 0; j < 784; j++) input[j] = byteInput[j] / 255d;

					for (int nnIndex = 0; nnIndex < nns.Count; nnIndex++) {
						NeuralNetwork nn = nns[nnIndex];

						nn.PutData(input);
						nn.Run();

						int answer = nn.GetMaxIndexInOutput();
						if (answer == digit) numOfGood[nnIndex]++;
						
						memory[nnIndex].Add(new KeyValuePair<int, int>(answer, digit));
					}
				}
				Console.Write(".");
			}
			Console.WriteLine(" Done.");
			
			Console.Write("Writing data to excel... ");
			WriteToExcel(numOfGood, memory);
		}

		private static int GetNextByte(Stream fileStream) {
			byte[] digit = new byte[1];
			fileStream.Read(digit, 0, 1);
			return digit[0];
		}

		private static void WriteToExcel(List<int> numOfGood, List<List<KeyValuePair<int, int>>> memory) {
			ExcelPackage.LicenseContext = LicenseContext.NonCommercial;
			using ExcelPackage package = new ExcelPackage(new FileInfo("results.xlsx"));
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
					KeyValuePair<int, int> pair = memory[i][index];

					worksheet.Cells[4 + index, currentColumn].Value = pair.Key;
					worksheet.Cells[4 + index, currentColumn + 1].Value = pair.Value;
				}

				worksheet.Column(currentColumn + 1).Style.Border.Right.Style = ExcelBorderStyle.Thin;
			}
			
			package.Save();
		}
	}
}