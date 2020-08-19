using System.Collections.Generic;
using NeuralNetworks.Misc;
using NeuralNetworks.Units;
using Newtonsoft.Json.Linq;

namespace NeuralNetworks.Layers {

public class SimpleLayer : Layer {
	public override EList<Unit> input { get; }
	public override EList<Unit> output { get; }

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

	public string Serialize() {
		JObject layer = new JObject {["type"] = nameof(SimpleLayer)};

		JArray unitsArray = new JArray();
		foreach (Unit unit in input) unitsArray.Add(unit.toJObject());
		layer["units"] = unitsArray;

		return layer.ToString();
	}

	public override void fillWeightsRandom() { }
	public override void fillBiasesRandom() { }
	public override void applyDerivativesToWeights(double learningFactor) { }
	public override void applyDerivativesToBiases(double learningFactor) { }
}

}