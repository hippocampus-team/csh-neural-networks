using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NeuralNetworks;
using NeuralNetworks.Misc;

namespace Testing {

internal static class Mnist {
	private const int PARALLEL_NNS = 4;
	private const int TRAIN_SIZE = 10;
	private const int TEST_SIZE = 4;
	
	private static string experimentTitle;

	public static void run() {
		Console.Write("Hello. Enter experiment title: ");
		experimentTitle = Console.ReadLine();
		string rootPath = $"./experiments/{experimentTitle}";

		Console.Write("Initialisation of NNs...");
		List<NeuralNetwork> nns = setupNNs();
		Console.WriteLine(" Done.");

		Console.WriteLine("Training started!");
		Console.Write("In progress");
		train(nns, TRAIN_SIZE);
		Console.WriteLine(" Done.");

		Console.WriteLine("Testing started!");
		Console.Write("In progress");
		NNsData testData = test(nns, TEST_SIZE);
		Console.WriteLine(" Done.");

		Console.Write("Writing data to excel... ");
		Utils.writeToExcel(testData, rootPath);
		Console.WriteLine(" Done.");

		Console.Write("Writing NNs configurations to files... ");
		Utils.writeNNsToFiles(nns, rootPath);
		Console.WriteLine(" Done.");

		Console.WriteLine("Process is completed and result is saved in excel file.");
		Console.ReadKey();
	}

	private static List<NeuralNetwork> setupNNs() {
		List<NeuralNetwork> nns = new List<NeuralNetwork>();
		for (int i = 0; i < PARALLEL_NNS; i++) {
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

		return nns;
	}

	public static void train(List<NeuralNetwork> nns, int iterations) {
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
				int digit = Utils.getNextByte(trainLabels);

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

	public static NNsData test(List<NeuralNetwork> nns, int iterations) {
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
				int digit = Utils.getNextByte(testLabels);

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
}

public class NNsData {
	public List<int> numOfGood { get; }
	public List<List<KeyValuePair<int, int>>> memory { get; }

	public NNsData(List<int> numOfGood, List<List<KeyValuePair<int, int>>> memory) {
		this.numOfGood = numOfGood;
		this.memory = memory;
	}
}

}