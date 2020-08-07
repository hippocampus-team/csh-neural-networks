using System;
using NeuralNetworks;
using NeuralNetworks.Misc;

namespace Testing {
	internal static class Autocoding {
		private const int testNum = 128;

		public static void Run() {
			NeuralNetwork nn = new NeuralNetwork(true);

			nn.SetInputLength(testNum);
			nn.AddDenceLayer(16);
			nn.AddDenceLayer(testNum);

			nn.FillRandomWeights();
			nn.FillRandomBiases();

			Random random = new Random(Guid.NewGuid().GetHashCode());
			EList<double> data = new EList<double>();
			for (int i = 0; i < testNum; i++) data.Add(0);

			for (int i = 0; i < 1000; i++) {
				for (int j = 0; j < testNum; j++) data[j] = Math.Round(random.NextDouble());

				nn.PutData(data);
				nn.Run();
				nn.Backpropagate(data, 1);

				Console.WriteLine($"Iteration {i} has cost {nn.GetTotalCost(data.ToList())}");
			}
		}
	}
}