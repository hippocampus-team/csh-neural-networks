using System.Collections.Generic;
using NeuralNetworks.Misc;
using NeuralNetworks.Units;

namespace NeuralNetworks.Layers {
	public class SimpleLayer : ILayer {
		public EList<Unit> input { get; }
		public EList<Unit> output { get; }

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

		public SimpleLayer(ILayer inputLayer) {
			EList<Unit> nodes = new EList<Unit>();

			foreach (Unit unit in inputLayer.output)
				nodes.Add(new ReferNode(unit));

			input = nodes;
			output = nodes;
		}

		public void Count() {
			foreach (Unit unit in output)
				((Node) unit).Count();
		}

		public void CountDerivatives() {
			foreach (Unit unit in input)
				unit.CountDerivatives();
		}

		public void CountDerivatives(EList<double> expectedOutput) {
			for (int i = 0; i < output.Count; i++)
				output[i].derivative = expectedOutput[i] - output[i].value;
		}

		public EList<double> GetInputValues() {
			EList<double> values = new EList<double>();

			foreach (Unit unit in input)
				values.Add(unit.value);

			return values;
		}

		public EList<double> GetOutputValues() {
			EList<double> values = new EList<double>();

			foreach (Unit unit in output)
				values.Add(unit.value);

			return values;
		}

		public void FillWeightsRandom()                              { }
		public void FillBiasesRandom()                               { }
		public void ApplyDerivativesToWeights(double learningFactor) { }
		public void ApplyDerivativesToBiases(double learningFactor)  { }
	}
}