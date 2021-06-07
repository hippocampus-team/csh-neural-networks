using System;
using System.Collections.Generic;
using System.IO;
using NeuralNetworks;

namespace Testing {

public class Deserializer {
	private const int testSize = 10;
	
	public static void run() {
		Console.Write("Hello. Enter experiments title: ");
		string experimentTitle = Console.ReadLine();
		
		Console.Write("Enter experiments description: ");
		string experimentDescription = Console.ReadLine();
		
		string rootPath = $"./experiments/{experimentTitle}";
		ExperimentLog log = new ExperimentLog(experimentTitle, experimentDescription, 1);
		
		Console.Write("Enter config files name: ./experiments/loads/");
		string loadName = Console.ReadLine();

		Console.Write("Deserialization of NN...");
		NeuralNetwork nn = deserialize($"./experiments/loads/{loadName}", log);
		Console.WriteLine(" Done.");

		Console.WriteLine("Testing started!");
		Console.Write("In progress");
		NNsTestResults<int> testResults = Mnist.test(new List<NeuralNetwork> {nn}, testSize, log);
		Console.WriteLine(" Done.");
		
		Console.Write("Saving log... ");
		Utils.endLogAndWriteToFile(log, rootPath);
		Console.WriteLine(" Done.");

		Console.Write("Writing data to excel... ");
		Utils.writeToExcel(testResults, rootPath);
		Console.WriteLine(" Done.");

		Console.WriteLine("Process is completed and result is saved in excel file.");
		Console.ReadKey();
	}

	private static NeuralNetwork deserialize(string path, ExperimentLog? log = null) {
		log?.startPhase("Deserialization", $"source file {path}", 0);
		string json = File.ReadAllText(path);
		NeuralNetwork nn = NeuralNetwork.deserialize(json);
		log?.endPhase();
		
		log?.recordTopologyFromNetwork(nn);
		return nn;
	}
}

}