using System;
using NeuralNetworks.Misc;
using NeuralNetworks.Units;

namespace NeuralNetworks.Layers {

public class DenceLayer : Layer {
	public sealed override EList<Unit> input { get; protected set; }
	public sealed override EList<Unit> output { get; protected set; }

	private DenceLayer() { }

	public DenceLayer(int n, Layer inputLayer, ActivationFunction activationFunction) {
		EList<Unit> neurons = new EList<Unit>();

		for (int i = 0; i < n; i++)
			neurons.Add(new Neuron(inputLayer.output, activationFunction));

		input = neurons;
		output = neurons;
	}

	public override void count() {
		foreach (Unit unit in output)
			unit.count();
	}

	public override void fillWeightsRandom() {
		Random rnd = new Random(Guid.NewGuid().GetHashCode());

		foreach (Unit unit in output) {
			Neuron neuron = (Neuron) unit;
			for (int i = 0; i < neuron.weights.Count; i++)
				neuron.weights[i] = (rnd.NextDouble() - 0.5) * Constants.WEIGHT_RANDOM_FILL_SPREAD;
		}
	}

	public override void fillBiasesRandom() {
		Random rnd = new Random(Guid.NewGuid().GetHashCode());

		foreach (Unit unit in output) {
			Neuron neuron = (Neuron) unit;
			neuron.bias = (rnd.NextDouble() - 0.5) * Constants.BIAS_RANDOM_FILL_SPREAD;
		}
	}

	public override void countDerivatives() {
		foreach (Unit unit in output)
			unit.countDerivatives();
	}

	public override void countDerivatives(EList<double> expectedOutput) {
		for (int i = 0; i < output.Count; i++)
			output[i].derivative = expectedOutput[i] - output[i].value;
	}

	public override void applyDerivativesToWeights(double learningFactor) {
		foreach (Unit unit in output)
			unit.applyDerivativesToWeights(learningFactor);
	}

	public override void applyDerivativesToBiases(double learningFactor) {
		foreach (Unit unit in output)
			unit.applyDerivativesToBias(learningFactor);
	}

	public override EList<double> getInputValues() {
		EList<double> values = new EList<double>();

		foreach (Unit unit in input)
			values.Add(unit.value);

		return values;
	}

	public override EList<double> getOutputValues() {
		EList<double> values = new EList<double>();

		foreach (Unit unit in output)
			values.Add(unit.value);

		return values;
	}
	public static DenceLayer getEmpty() => new DenceLayer();
}

}