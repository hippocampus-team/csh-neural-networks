using System;
using System.Collections.Generic;
using NeuralNetworks.Misc;
using NeuralNetworks.Units;
using Newtonsoft.Json.Linq;

namespace NeuralNetworks.Layers {

public class ConvolutionalLayer : SameInputOutputLayer {
	public override IEnumerable<Unit> units => neurons;

	private readonly DepthList<ConvolutionalNeuron> neurons;
	private List<List<Filter>> kernels { get; }
	private MatrixModel model { get; }

	private ConvolutionalLayer() => throw new NotImplementedException();

	public ConvolutionalLayer(LayerConnection inputConnection, Filter filter, int filtersAmount, int stride, 
							  ActivationFunction activationFunction) {
		kernels = new List<List<Filter>>();
		model = new MatrixModel(inputConnection.length, stride);

		for (int f = 0; f < filtersAmount; f++) {
			kernels.Add(new List<Filter>());

			for (int d = 0; d < inputConnection.depth; d++)
				kernels[f].Add((Filter) filter.Clone());
		}

		neurons = new DepthList<ConvolutionalNeuron>(filtersAmount * inputConnection.depth);

		for (int f = 0; f < filtersAmount; f++)
		for (int d = 0; d < inputConnection.depth; d++)
		for (int o = 0; o < model.filterOutputsCount(filter); o++) {
			List<int> indexes = new List<int>();

			for (int x = 0; x < filter.count; x++) {
				int inner = x % filter.size + x / filter.size * model.size;
				int outer = o % model.filterLineCount(filter) + o / model.filterLineCount(filter) * model.size;

				indexes.Add(inner + outer);
			}
			
			List<Unit> inputUnits = new List<Unit>(inputConnection.length);

			for (int i = inputConnection.depth * inputConnection.length; 
				 i < (inputConnection.depth + 1) * inputConnection.length; i++) 
				inputUnits.Add(inputConnection[i]);

			neurons.Add(new ConvolutionalNeuron(inputUnits, kernels[f][d], indexes, activationFunction));
		}

		input = new ConvolutionalLayerConnection(neurons);
	}

	public override void count() {
		foreach (ConvolutionalNeuron neuron in neurons)
			neuron.count();
	}

	public override void fillParametersRandomly() {
		Random rnd = new Random(Guid.NewGuid().GetHashCode());

		for (int i = 0; i < kernels.Count; i++)
		for (int j = 0; j < kernels[i].Count; j++)
		for (int k = 0; k < kernels[i][j].values.Count; k++)
			kernels[i][j].values[k] = (rnd.NextDouble() - 0.5) * Constants.weightRandomFillSpread;
		
		foreach (ConvolutionalNeuron neuron in neurons)
			neuron.bias = (rnd.NextDouble() - 0.5) * Constants.biasRandomFillSpread;
	}

	public override void countDerivativesOfPreviousLayer() {
		foreach (ConvolutionalNeuron neuron in neurons)
			neuron.countDerivativesOfInputUnits();
	}

	public override void countDerivatives(List<double> expectedOutput) { }

	public override void applyDerivativesToParameters(double learningFactor) {
		foreach (ConvolutionalNeuron neuron in neurons)
			neuron.applyDerivativeToParameters(learningFactor);
	}

	public override List<double> getInputValues() => throw new NotImplementedException();
	public override Layer fillFromJObject(JObject json) => throw new NotImplementedException();

	public static ConvolutionalLayer getEmpty() => new ConvolutionalLayer();
	
	private class ConvolutionalLayerConnection : LayerConnection {
		private readonly DepthList<ConvolutionalNeuron> neurons;
		
		public IEnumerable<Unit> enumerable => neurons.toList();

		public Unit this[int index] {
			get => neurons[index];
			set => neurons[index] = (ConvolutionalNeuron) value;
		}
		
		public Unit this[int index, int depthIndex] {
			get => neurons[index, depthIndex];
			set => neurons[index, depthIndex] = (ConvolutionalNeuron) value;
		}

		public int length => neurons.length;
		public int depth => neurons.depth;

		public ConvolutionalLayerConnection(DepthList<ConvolutionalNeuron> neurons) {
			this.neurons = neurons;
		}
	}
}

}