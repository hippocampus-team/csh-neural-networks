using System;
using NeuralNetworks.Misc;

namespace NeuralNetworks {

public class DenceLayer : ILayer {
	public EList<Unit> Input { get; set; }
	public EList<Unit> Output { get; set; }
	
	public DenceLayer(int n, ILayer inputLayer) {
		EList<Unit> neurons = new EList<Unit>();

		for (int i = 0; i < n; i++)
			neurons.Add(new Neuron(inputLayer.Output));

		Input = neurons;
		Output = neurons;
	}

	//public void MutateRandomWeights()
	//{
	//    Random rnd = new Random(Guid.NewGuid().GetHashCode());

	//    int n = rnd.Next(Output.Count);
	//    Neuron neuron = (Neuron)Output[n];

	//    int w = rnd.Next(neuron.Weights.Count);
	//    neuron.Weights[w] += (rnd.NextDouble() - 0.5) * Constants.weightMutationSpread;

	//    neuron.Weights[w] = Tools.Clamp(neuron.Weights[w], Constants.weightMutatedMin, Constants.weightMutatedMax);
	//}

	//public void MutateRandomBiases()
	//{
	//    Random rnd = new Random(Guid.NewGuid().GetHashCode());

	//    int n = rnd.Next(Output.Count);
	//    Neuron neuron = (Neuron)Output[n];

	//    neuron.Bias += (rnd.NextDouble() - 0.5) * Constants.biasMutationSpread;

	//    neuron.Bias = Tools.Clamp(neuron.Bias, Constants.biasMutatedMin, Constants.biasMutatedMax);
	//}

	public void Count() {
		foreach (Unit unit in Output)
			unit.Count();
	}

	public void FillWeightsRandom() {
		Random rnd = new Random(Guid.NewGuid().GetHashCode());

		foreach (Unit unit in Output) {
			Neuron neuron = (Neuron) unit;
			for (int i = 0; i < neuron.Weights.Count; i++)
				neuron.Weights[i] = (rnd.NextDouble() - 0.5) * Constants.weightRandomFillSpread;
		}
	}

	public void FillBiasesRandom() {
		Random rnd = new Random(Guid.NewGuid().GetHashCode());

		foreach (Unit unit in Output) {
			Neuron neuron = (Neuron) unit;
			neuron.Bias = (rnd.NextDouble() - 0.5) * Constants.biasRandomFillSpread;
		}
	}

	public void CountDerivatives() {
		foreach (Unit unit in Output)
			unit.CountDerivatives();
	}

	public void CountDerivatives(EList<double> expectedOutput) {
		for (int i = 0; i < Output.Count; i++)
			Output[i].Derivative = expectedOutput[i] - Output[i].Value;
	}

	public void ApplyDerivativesToWeights(double learningFactor) {
		foreach (Unit unit in Output)
			unit.ApplyDerivativesToWeights(learningFactor);
	}

	public void ApplyDerivativesToBiases(double learningFactor) {
		foreach (Unit unit in Output)
			unit.ApplyDerivativesToBias(learningFactor);
	}
	
	public EList<double> GetInputValues() {
		EList<double> values = new EList<double>();

		foreach (Unit unit in Input)
			values.Add(unit.Value);

		return values;
	}
	public EList<double> GetOutputValues() {
		EList<double> values = new EList<double>();

		foreach (Unit unit in Output)
			values.Add(unit.Value);

		return values;
	}
}

}