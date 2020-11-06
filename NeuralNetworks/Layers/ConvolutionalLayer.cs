using System;
using System.Collections.Generic;
using NeuralNetworks.Misc;
using NeuralNetworks.Units;

namespace NeuralNetworks.Layers {

public class ConvolutionalLayer : Layer {
	public sealed override EList<Unit> input { get; protected set; }
	public sealed override EList<Unit> output { get; protected set; }
	public override IEnumerable<Unit> neurons => input;
	
	private List<List<Filter>> kernels { get; }
	private MatrixModel model { get; }

	private ConvolutionalLayer() { }

	public ConvolutionalLayer(Layer inputLayer, Filter filter, int filtersAmount, int stride, 
							  ActivationFunction activationFunction) {
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
		for (int o = 0; o < model.filterOutputsCount(filter); o++) {
			List<int> indexes = new List<int>();

			for (int x = 0; x < filter.count; x++) {
				int inner = x % filter.size + x / filter.size * model.size;
				int outer = o % model.filterLineCount(filter) + o / model.filterLineCount(filter) * model.size;

				indexes.Add(inner + outer);
			}

			neurons.Add(new ConvolutionalNeuron(inputLayer.output, kernels[f][c], indexes, c, activationFunction));
		}

		input = neurons;
		output = neurons;
	}

	public override void count() {
		foreach (Unit unit in output)
			unit.count();
	}

	public override void fillWeightsRandom() {
		Random rnd = new Random(Guid.NewGuid().GetHashCode());

		for (int i = 0; i < kernels.Count; i++)
		for (int j = 0; j < kernels[i].Count; j++)
		for (int k = 0; k < kernels[i][j].values.Count; k++)
			kernels[i][j].values[k] = (rnd.NextDouble() - 0.5) * Constants.WEIGHT_RANDOM_FILL_SPREAD;
	}

	public override void fillBiasesRandom() {
		Random rnd = new Random(Guid.NewGuid().GetHashCode());

		foreach (Unit unit in output) {
			ConvolutionalNeuron neuron = (ConvolutionalNeuron) unit;
			neuron.bias = (rnd.NextDouble() - 0.5) * Constants.BIAS_RANDOM_FILL_SPREAD;
		}
	}

	public override void countDerivatives() {
		foreach (Unit unit in output)
			unit.countDerivatives();
	}

	public override void countDerivatives(EList<double> expectedOutput) {
		for (int r = 0; r < output.rows; r++)
		for (int c = 0; c < output.columns; c++)
			output[r, c].derivative = expectedOutput[r, c] - output[r, c].value;
	}

	public override void applyDerivativesToWeights(double learningFactor) {
		foreach (Unit unit in output)
			unit.applyDerivativesToWeights(learningFactor);
	}

	public override void applyDerivativesToBiases(double learningFactor) {
		foreach (Unit unit in output)
			unit.applyDerivativesToBias(learningFactor);
	}

	public override EList<double> getInputValues() => throw new NotImplementedException();
	public override EList<double> getOutputValues() => throw new NotImplementedException();
	public static ConvolutionalLayer getEmpty() => new ConvolutionalLayer();
}

}