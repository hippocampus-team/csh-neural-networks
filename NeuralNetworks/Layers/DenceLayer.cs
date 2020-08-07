using System;
using NeuralNetworks.Misc;
using NeuralNetworks.Units;

namespace NeuralNetworks.Layers {
	public class DenceLayer : Layer {
		public override EList<Unit> input { get; }
		public override EList<Unit> output { get; }

		public DenceLayer(int n, Layer inputLayer) {
			EList<Unit> neurons = new EList<Unit>();

			for (int i = 0; i < n; i++)
				neurons.Add(new Neuron(inputLayer.output));

			input = neurons;
			output = neurons;
		}

		public override void Count() {
			foreach (Unit unit in output)
				unit.Count();
		}

		public override void FillWeightsRandom() {
			Random rnd = new Random(Guid.NewGuid().GetHashCode());

			foreach (Unit unit in output) {
				Neuron neuron = (Neuron) unit;
				for (int i = 0; i < neuron.weights.Count; i++)
					neuron.weights[i] = (rnd.NextDouble() - 0.5) * Constants.weightRandomFillSpread;
			}
		}

		public override void FillBiasesRandom() {
			Random rnd = new Random(Guid.NewGuid().GetHashCode());

			foreach (Unit unit in output) {
				Neuron neuron = (Neuron) unit;
				neuron.bias = (rnd.NextDouble() - 0.5) * Constants.biasRandomFillSpread;
			}
		}

		public override void CountDerivatives() {
			foreach (Unit unit in output)
				unit.CountDerivatives();
		}

		public override void CountDerivatives(EList<double> expectedOutput) {
			for (int i = 0; i < output.Count; i++)
				output[i].derivative = expectedOutput[i] - output[i].value;
		}

		public override void ApplyDerivativesToWeights(double learningFactor) {
			foreach (Unit unit in output)
				unit.ApplyDerivativesToWeights(learningFactor);
		}

		public override void ApplyDerivativesToBiases(double learningFactor) {
			foreach (Unit unit in output)
				unit.ApplyDerivativesToBias(learningFactor);
		}

		public override EList<double> GetInputValues() {
			EList<double> values = new EList<double>();

			foreach (Unit unit in input)
				values.Add(unit.value);

			return values;
		}

		public override EList<double> GetOutputValues() {
			EList<double> values = new EList<double>();

			foreach (Unit unit in output)
				values.Add(unit.value);

			return values;
		}
	}
}