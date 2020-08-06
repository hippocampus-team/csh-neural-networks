using System;
using System.Collections.Generic;
using NeuralNetworks.Misc;
using NeuralNetworks.Units;

namespace NeuralNetworks.Layers {
	public class ConvolutionalLayer : ILayer {
		private List<List<Filter>> kernels { get; }
		private MatrixModel model { get; }

		public EList<Unit> input { get; }
		public EList<Unit> output { get; }

		public ConvolutionalLayer(ILayer inputLayer, Filter filter, int filtersAmount, int stride) {
			kernels = new List<List<Filter>>();
			model = new MatrixModel(inputLayer.output, stride);

			for (int f = 0; f < filtersAmount; f++) {
				kernels.Add(new List<Filter>());

				for (int c = 0; c < inputLayer.output.columns; c++)
					kernels[f].Add((Filter) filter.Clone());
			}

			EList<Unit> neurons = new EList<Unit>(filtersAmount * inputLayer.output.columns);

			for (int f = 0; f < filtersAmount; f++)
			for (int c = 0; c < inputLayer.output.columns; c++)
			for (int o = 0; o < model.FilterOutputsCount(filter); o++) {
				List<int> indexes = new List<int>();

				for (int x = 0; x < filter.Count(); x++) {
					int inner = x % filter.size + x / filter.size * model.size;
					int outer = o % model.FilterLineCount(filter) + o / model.FilterLineCount(filter) * model.size;

					indexes.Add(inner + outer);
				}

				neurons.Add(new ConvolutionalNeuron(inputLayer.output, kernels[f][c], indexes, c));
			}

			input = neurons;
			output = neurons;
		}

		public void Count() {
			foreach (Unit unit in output)
				unit.Count();
		}

		public void FillWeightsRandom() {
			Random rnd = new Random(Guid.NewGuid().GetHashCode());

			for (int i = 0; i < kernels.Count; i++)
			for (int j = 0; j < kernels[i].Count; j++)
			for (int k = 0; k < kernels[i][j].values.Count; k++)
				kernels[i][j].values[k] = (rnd.NextDouble() - 0.5) * Constants.weightRandomFillSpread;
		}

		public void FillBiasesRandom() {
			Random rnd = new Random(Guid.NewGuid().GetHashCode());

			foreach (Unit unit in output)
				unit.bias = (rnd.NextDouble() - 0.5) * Constants.biasRandomFillSpread;
		}

		public void CountDerivatives() {
			foreach (Unit unit in output)
				unit.CountDerivatives();
		}

		public void CountDerivatives(EList<double> expectedOutput) {
			for (int r = 0; r < output.rows; r++)
			for (int c = 0; c < output.columns; c++)
				output[r, c].derivative = expectedOutput[r, c] - output[r, c].value;
		}

		public void ApplyDerivativesToWeights(double learningFactor) {
			foreach (Unit unit in output)
				unit.ApplyDerivativesToWeights(learningFactor);
		}

		public void ApplyDerivativesToBiases(double learningFactor) {
			foreach (Unit unit in output)
				unit.ApplyDerivativesToBias(learningFactor);
		}

		public EList<double> GetInputValues()  => throw new NotImplementedException();
		public EList<double> GetOutputValues() => throw new NotImplementedException();
	}
}