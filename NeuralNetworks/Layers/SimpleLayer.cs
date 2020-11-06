using System.Collections.Generic;
using NeuralNetworks.Misc;
using NeuralNetworks.Units;

namespace NeuralNetworks.Layers {

public class SimpleLayer : Layer {
	public sealed override EList<Unit> input { get; protected set; }
	public sealed override EList<Unit> output { get; protected set; }
	public override IEnumerable<Unit> neurons => input;

	public SimpleLayer(int n) {
		EList<Unit> nodes = new EList<Unit>();

		for (int i = 0; i < n; i++)
			nodes.Add(new Node());

		input = nodes;
		output = nodes;
	}

	public SimpleLayer(List<double> values) {
		EList<Unit> nodes = new EList<Unit>();

		foreach (double value in values)
			nodes.Add(new Node(value));

		input = nodes;
		output = nodes;
	}

	public SimpleLayer(Layer inputLayer) {
		EList<Unit> nodes = new EList<Unit>();

		foreach (Unit unit in inputLayer.output)
			nodes.Add(new ReferNode(unit));

		input = nodes;
		output = nodes;
	}

	public override void count() {
		foreach (Unit unit in output)
			((Node) unit).count();
	}

	public override void countDerivatives() {
		foreach (Unit unit in input)
			unit.countDerivatives();
	}

	public override void countDerivatives(EList<double> expectedOutput) {
		for (int i = 0; i < output.Count; i++)
			output[i].derivative = expectedOutput[i] - output[i].value;
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

	public override void fillWeightsRandom() { }
	public override void fillBiasesRandom() { }
	public override void applyDerivativesToWeights(double learningFactor) { }
	public override void applyDerivativesToBiases(double learningFactor) { }
	public static SimpleLayer getEmpty() => new SimpleLayer(0);
}

}