using System;
using System.Collections.Generic;
using NeuralNetworks;
using NeuralNetworks.Misc;

namespace Testing {

internal static class Autocoding {
	private const int testNum = 128;

	public static void run() {
		NeuralNetwork nn = new NeuralNetwork();

		nn.setInputLength(testNum);
		nn.addDenseLayer(16, new Sigmoid());
		nn.addDenseLayer(testNum, new Sigmoid());

		nn.fillPropertiesRandomly();

		Random random = new Random(Guid.NewGuid().GetHashCode());
		List<double> data = new List<double>();
		for (int i = 0; i < testNum; i++) data.Add(0);

		for (int i = 0; i < 1000; i++) {
			for (int j = 0; j < testNum; j++) data[j] = Math.Round(random.NextDouble());

			nn.putData(data);
			nn.run();
			nn.backpropagate(data, 1);

			Console.WriteLine($"Iteration {i} has cost {nn.getTotalCost(data)}");
		}
	}
}

}