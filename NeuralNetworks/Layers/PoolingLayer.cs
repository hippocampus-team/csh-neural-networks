using System;
using NeuralNetworks.Misc;
using NeuralNetworks.Units;

namespace NeuralNetworks.Layers {
	public class PoolingLayer : ILayer {
		private MatrixModel model { get; }
		private Filter mask { get; }

		public EList<Unit> input { get; }
		public EList<Unit> output { get; }

		public PoolingLayer(ILayer inputLayer, Filter mask, int stride, PoolingNode.PoolingMethod method) {
			model = new MatrixModel(inputLayer.output, stride);
			this.mask = mask;

			EList<Unit> nodes = new EList<Unit>(inputLayer.output.columns);

			for (int c = 0; c < inputLayer.output.columns; c++)
			for (int i = 0; i < model.FilterOutputsCount(mask); i++) {
				EList<Unit> inputUnits = new EList<Unit>();

				for (int x = 0; x < mask.Count(); x++) {
					int inner = x % mask.size + x / this.mask.size * model.size;
					int outer = i % model.FilterLineCount(this.mask) +
						i / model.FilterLineCount(this.mask) * model.size;

					inputUnits.Add(inputLayer.output[inner + outer, c]);
				}

				nodes.Add(new PoolingNode(inputUnits, method));
			}

			input = nodes;
			output = nodes;
		}

		public void Count() {
			foreach (Unit unit in output)
				unit.Count();
		}

		public void CountDerivatives() {
			double part = 1d / mask.Count();

			foreach (Unit node in input)
			foreach (Unit inputLayerUnit in node.inputUnits)
				inputLayerUnit.derivative += part * node.derivative;
		}

		public void CountDerivatives(EList<double> expectedOutput) {
			for (int r = 0; r < output.rows; r++)
			for (int c = 0; c < output.columns; c++)
				output[r, c].derivative = expectedOutput[r, c] - output[r, c].value;
		}

		public EList<double> GetInputValues() => throw new NotImplementedException();

		public EList<double> GetOutputValues() => throw new NotImplementedException();

		public void FillWeightsRandom()                              { }
		public void FillBiasesRandom()                               { }
		public void ApplyDerivativesToWeights(double learningFactor) { }
		public void ApplyDerivativesToBiases(double learningFactor)  { }
	}
}