using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NeuralNetworks;
using NeuralNetworks.Misc;

namespace Testing {

internal static class Mnist {
	private const int parallelNns = 2;
	private const int trainSize = 1;
	private const int testSize = 3;
	private const double learningStep = 0.1d;

	private static ExperimentLog log;

	public static void run() {
		Console.Write("Hello. Enter experiments title: ");
		string experimentTitle = Console.ReadLine();
		
		Console.Write("Enter experiments description: ");
		string experimentDescription = Console.ReadLine();
		
		string rootPath = $"./experiments/{experimentTitle}";
		log = new ExperimentLog(experimentTitle, experimentDescription);

		Console.Write("Initialisation of NNs...");
		List<NeuralNetwork> nns = setupNNs();
		Console.WriteLine(" Done.");

		Console.WriteLine("Training started!");
		Console.Write("In progress");
		train(nns, trainSize);
		Console.WriteLine(" Done.");

		Console.WriteLine("Testing started!");
		Console.Write("In progress");
		NNsTestResults<int> testResults = test(nns, testSize);
		Console.WriteLine(" Done.");
		
		Console.Write("Saving log... ");
		Utils.endLogAndWriteToFile(log, rootPath);
		Console.WriteLine(" Done.");

		Console.Write("Writing results to excel... ");
		Utils.writeToExcel(testResults, rootPath);
		Console.WriteLine(" Done.");

		Console.Write("Writing NNs configurations to files... ");
		Utils.writeNNsToFiles(nns, rootPath);
		Console.WriteLine(" Done.");

		Console.WriteLine("Process is completed and result is saved in excel file.");
		Console.ReadKey();
	}

	private static List<NeuralNetwork> setupNNs() {
		List<NeuralNetwork> nns = new List<NeuralNetwork>();
		for (int i = 0; i < parallelNns; i++) {
			NeuralNetwork nn = new NeuralNetwork();

			nn.setInputLength(784);
			nn.addDenseLayer(300, new Sigmoid());
			nn.addDenseLayer(10, new Sigmoid());
			
			///// LeNet-5 ? 
			// nn.setInputLength(784);
			// nn.addPoolingLayer(new Filter(2), 2, PoolingNode.PoolingMethod.average);
			// nn.addConvolutionalLayer(new Filter(5), 16, 1);
			// nn.addPoolingLayer(new Filter(2), 2, PoolingNode.PoolingMethod.average);
			// nn.addDenseLayer(120);
			// nn.addDenseLayer(84);
			// nn.addDenseLayer(10);

			nn.fillRandomWeights();
			nn.fillRandomBiases();

			nns.Add(nn);
		}
		
		log.recordTopologyFromNetwork(nns[0]);

		return nns;
	}

	public static void train(List<NeuralNetwork> nns, int iterations) {
		FileStream trainImages = new FileStream("data/train_imgs", FileMode.Open);
		FileStream trainLabels = new FileStream("data/train_lbls", FileMode.Open);
		trainImages.Read(new byte[4 * 4], 0, 4 * 4);
		trainLabels.Read(new byte[4 * 2], 0, 4 * 2);

		EList<double> answer = new EList<double>();
		for (int i = 0; i < 10; i++) answer.Add(0);

		EList<double> input = new EList<double>();
		for (int i = 0; i < 784; i++) input.Add(0);
		
		log.startPhase("Training", 
					   $"MNIST dataset, learning step {learningStep}, correct answer as 1, wrong answer as 0", 
					   iterations * 1000, parallelNns);

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
				foreach (NeuralNetwork nn in nns) nn.backpropagate(answer, learningStep);
				answer[digit] = 0;
			}
			Console.Write(".");
		}
		
		log.endPhase();
	}

	public static NNsTestResults<int> test(List<NeuralNetwork> nns, int iterations) {
		FileStream testImages = new FileStream("data/test_imgs", FileMode.Open);
		FileStream testLabels = new FileStream("data/test_lbls", FileMode.Open);
		testImages.Read(new byte[4 * 4], 0, 4 * 4);
		testLabels.Read(new byte[4 * 2], 0, 4 * 2);

		EList<double> input = new EList<double>();
		for (int i = 0; i < 784; i++) input.Add(0);

		List<int> correctAnswersAmount = nns.Select(nn => 0).ToList();
		List<int> correctAnswers = new List<int>(iterations * 1000);
		List<List<int>> answers = nns.Select(nn => new List<int>()).ToList();
		
		log.startPhase("Testing", 
					   $"MNIST dataset", 
					   iterations * 1000, parallelNns);

		for (int h = 0; h < iterations; h++) {
			for (int i = 0; i < 1000; i++) {
				int digit = Utils.getNextByte(testLabels);
				correctAnswers.Add(digit);

				byte[] byteInput = new byte[784];
				testImages.Read(byteInput, 0, 784);
				for (int j = 0; j < 784; j++) input[j] = byteInput[j] / 255d;

				for (int nnIndex = 0; nnIndex < nns.Count; nnIndex++) {
					NeuralNetwork nn = nns[nnIndex];

					nn.putData(input);
					nn.run();

					int answer = nn.getMaxIndexInOutput();
					if (answer == digit) correctAnswersAmount[nnIndex]++;

					answers[nnIndex].Add(answer);
				}
			}
			Console.Write(".");
		}
		
		log.endPhase();

		return new NNsTestResults<int>(correctAnswersAmount, correctAnswers, answers);
	}
}

}