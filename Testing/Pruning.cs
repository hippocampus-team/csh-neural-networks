using System;
using System.Collections.Generic;
using NeuralNetworks;
using NeuralNetworks.Misc;
using NeuralNetworks.Units;

namespace Testing {

public class Pruning {
	private const int testNum = 64;
		
	public static void run() {
		FlexibleNeuralNetwork nn = new FlexibleNeuralNetwork();

		nn.setInputLength(testNum);
		nn.addDenseLayer(20, new ModifiedSigmoid());
		nn.addDenseLayer(testNum, new ModifiedSigmoid());

		nn.fillPropertiesRandomly();

		Random random = new Random(Guid.NewGuid().GetHashCode());
		List<double> data = new List<double>();
		for (int i = 0; i < testNum; i++) data.Add(0);

		for (int i = 0; i < 100; i++) {
			for (int j = 0; j < testNum; j++) data[j] = Math.Round(random.NextDouble());

			nn.putData(data);
			nn.run();
			nn.backpropagate(data, 1);

			Console.WriteLine($"Iteration {i} has cost {nn.getTotalCost(data)}");
		}
		
		Console.WriteLine("/// Starting pruning");

		List<PrunableNeuron> trackList = new List<PrunableNeuron>();
		for (int i = 0; i < nn[1].input.length; i++)
			trackList.Add(nn.makeNeuronPrunable(1, i));
		
		for (int i = 0; i < 10; i++) {
			for (int p = 0; p < 64; p++) {
				for (int j = 0; j < testNum; j++) data[j] = Math.Round(random.NextDouble());

				nn.putData(data);
				nn.run();
				nn.backpropagate(data, 1);

				Console.WriteLine($"Iteration {i} has cost {nn.getTotalCost(data)}");
			}

			foreach (PrunableNeuron neuron in trackList) neuron.prune(10);
		}
	}
}

}