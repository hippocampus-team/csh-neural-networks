using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NeuralNetworks;
using NeuralNetworks.Misc;

namespace Testing {
	internal static class Mnist {
		private static List<NeuralNetwork> nns;
		
		public static void Run() {
			Console.Write("Initialisation of MNIST 784+300+10 test... ");
			SetupNNs();
			Console.WriteLine("Done");
			
			Console.WriteLine("Training started!");
			Train();

			Console.WriteLine("Testing started!");
			Test();
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
			for (int i = 0; i < 10; i++) answer.Add(0);
			
			EList<double> input = new EList<double>();
			for (int i = 0; i < 784; i++) input.Add(0);
			
			for (int h = 0; h < 10; h++) {
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
					foreach (NeuralNetwork nn in nns) nn.Backpropagate(answer, 10);
					answer[digit] = 0;
				}
				
				Console.WriteLine($"Train iteration {h * 1000}");
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

			for (int i = 0; i < 5000; i++) {
				int digit = GetNextByte(testLabels);
				
				byte[] byteInput = new byte[784];
				testImages.Read(byteInput, 0, 784);
				for (int j = 0; j < 784; j++) input[j] = byteInput[j] / 255d;

				for (int index = 0; index < nns.Count; index++) {
					NeuralNetwork nn = nns[index];
					
					nn.PutData(input);
					nn.Run();

					int maxIndex = nn.GetMaxIndexInOutput();
					if (maxIndex == digit) numOfGood[index]++;
				}

				// Console.WriteLine($"Iteration {i} has {digit[0]} {maxIndex}");
			}

			for (int index = 0; index < nns.Count; index++) 
				Console.WriteLine($"Results for {index} are: {numOfGood[index]} out of 5000 => {numOfGood[index] / 5000d}");
		}

		private static int GetNextByte(Stream fileStream) {
			byte[] digit = new byte[1];
			fileStream.Read(digit, 0, 1);
			return digit[0];
		}
	}
}