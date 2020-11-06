using System;
using System.Collections.Generic;
using System.IO;
using NeuralNetworks;

namespace Testing {

public class Deserializer {
	private const int TEST_SIZE = 4;
	
	public static void run() {
		Console.Write("Hello. Enter experiment title: ");
		string experimentTitle = Console.ReadLine();
		string experimentPath = $"./experiments/{experimentTitle}";
		
		Console.Write("Enter config files name: ./experiments/loads/");
		string loadName = Console.ReadLine();

		Console.Write("Deserialization of NN...");
		NeuralNetwork nn = readNN($"./experiments/loads/{loadName}");
		Console.WriteLine(" Done.");

		Console.WriteLine("Testing started!");
		Console.Write("In progress");
		NNsData<int> testData = Mnist.test(new List<NeuralNetwork> {nn}, TEST_SIZE);
		Console.WriteLine(" Done.");

		Console.Write("Writing data to excel... ");
		Utils.writeToExcel(testData, experimentPath);
		Console.WriteLine(" Done.");

		Console.WriteLine("Process is completed and result is saved in excel file.");
		Console.ReadKey();
	}

	private static NeuralNetwork readNN(string path) {
		string json = File.ReadAllText(path);
		return NeuralNetwork.deserialize(json);
	}
}

}