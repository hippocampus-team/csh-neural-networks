using System.Collections.Generic;
using NeuralNetworks.Misc;

namespace NeuralNetworks {

public class SimpleLayer : ILayer {
	public EList<Unit> Input { get; set; }
	public EList<Unit> Output { get; set; }
	
	public SimpleLayer(int n) {
		EList<Unit> nodes = new EList<Unit>();

		for (int i = 0; i < n; i++)
			nodes.Add(new Node());

		Input = nodes;
		Output = nodes;
	}
	
	public SimpleLayer(List<double> values) {
		EList<Unit> nodes = new EList<Unit>();

		foreach (double value in values)
			nodes.Add(new Node(value));

		Input = nodes;
		Output = nodes;
	}

	public SimpleLayer(ILayer inputLayer) {
		EList<Unit> nodes = new EList<Unit>();

		foreach (Unit unit in inputLayer.Output)
			nodes.Add(new ReferNode(unit));

		Input = nodes;
		Output = nodes;
	}

	public void Count() {
		foreach (Unit unit in Output) 
			((Node) unit).Count();
	}

	public void CountDerivatives() {
		foreach (Unit unit in Input)
			unit.CountDerivatives();
	}

	public void CountDerivatives(EList<double> expectedOutput) {
		for (int i = 0; i < Output.Count; i++)
			Output[i].Derivative = expectedOutput[i] - Output[i].Value;
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
	
	public void FillWeightsRandom() { }
	public void FillBiasesRandom() { }
	public void ApplyDerivativesToWeights(double learningFactor) { }
	public void ApplyDerivativesToBiases(double learningFactor) { }
}

}