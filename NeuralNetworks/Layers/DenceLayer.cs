using System;
using NeuralNetworks.Misc;
using NeuralNetworks.Units;

namespace NeuralNetworks.Layers {
	public class DenceLayer : ILayer {
		public EList<Unit> input { get; }
		public EList<Unit> output { get; }

		public DenceLayer(int n, ILayer inputLayer) {
			EList<Unit> neurons = new EList<Unit>();

			for (int i = 0; i < n; i++)
				neurons.Add(new Neuron(inputLayer.output));

			input = neurons;
			output = neurons;
		}

		public void Count() {
			foreach (Unit unit in output)
				unit.Count();
		}

		public void FillWeightsRandom() {
			Random rnd = new Random(Guid.NewGuid().GetHashCode());

			foreach (Unit unit in output) {
				Neuron neuron = (Neuron) unit;
				for (int i = 0; i < neuron.weights.Count; i++)
					neuron.weights[i] = (rnd.NextDouble() - 0.5) * Constants.weightRandomFillSpread;
			}
		}

		public void FillBiasesRandom() {
			Random rnd = new Random(Guid.NewGuid().GetHashCode());

			foreach (Unit unit in output) {
				Neuron neuron = (Neuron) unit;
				neuron.bias = (rnd.NextDouble() - 0.5) * Constants.biasRandomFillSpread;
			}
		}

		public void CountDerivatives() {
			foreach (Unit unit in output)
				unit.CountDerivatives();
		}

		public void CountDerivatives(EList<double> expectedOutput) {
			for (int i = 0; i < output.Count; i++)
				output[i].derivative = expectedOutput[i] - output[i].value;
		}

		public void ApplyDerivativesToWeights(double learningFactor) {
			foreach (Unit unit in output)
				unit.ApplyDerivativesToWeights(learningFactor);
		}

		public void ApplyDerivativesToBiases(double learningFactor) {
			foreach (Unit unit in output)
				unit.ApplyDerivativesToBias(learningFactor);
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
	}
}