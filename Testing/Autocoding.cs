using System;
using NeuralNetworks;
using NeuralNetworks.Misc;

namespace Testing {

internal static class Autocoding {
	private const int TEST_NUM = 128;

	public static void run() {
		NeuralNetwork nn = new NeuralNetwork();

		nn.setInputLength(TEST_NUM);
		nn.addDenseLayer(16, new Sigmoid());
		nn.addDenseLayer(TEST_NUM, new Sigmoid());

		nn.fillRandomWeights();
		nn.fillRandomBiases();

		Random random = new Random(Guid.NewGuid().GetHashCode());
		EList<double> data = new EList<double>();
		for (int i = 0; i < TEST_NUM; i++) data.Add(0);

		for (int i = 0; i < 1000; i++) {
			for (int j = 0; j < TEST_NUM; j++) data[j] = Math.Round(random.NextDouble());

			nn.putData(data);
			nn.run();
			nn.backpropagate(data, 1);

			Console.WriteLine($"Iteration {i} has cost {nn.getTotalCost(data.toList())}");
		}
	}
}

}