using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Layers;
using NeuralNetworks.Misc;
using NeuralNetworks.Units;
using Newtonsoft.Json.Linq;

namespace NeuralNetworks {

public class NeuralNetwork {
	private string name { get; }
	private string description { get; }
	private List<Layer> layers { get; }
	private bool useBiases { get; }

	public Layer this[int layer] => layers[layer];
	
	public NeuralNetwork() : this("NeuralNetwork") { }

	public NeuralNetwork(string name) {
		this.name = name;
		description = "";
		layers = new List<Layer>();
		useBiases = true;
	}

	public void run() {
		foreach (Layer layer in layers)
			layer.count();
	}

	public double getCost(double expectedOutput, int index) =>
		1 - Math.Pow(expectedOutput - layers.Last().output[index].value, 2);

	public IEnumerable<double> getCosts(IEnumerable<double> expectedOutput) =>
		expectedOutput.Select((value, i) => 1 - Math.Pow(value - layers.Last().output[i].value, 2)).ToList();

	public double getTotalCost(IEnumerable<double> expectedOutput) => getCosts(expectedOutput).Average();

	public int getMaxIndexInOutput() {
		EList<double> outputs = layers.Last().getOutputValues();

		int index = 0;
		for (int i = 1; i < outputs.Count; i++) {
			if (outputs[i] > outputs[index])
				index = i;
		}
		return index;
	}

	public void fillRandomWeights() {
		foreach (Layer layer in layers)
			layer.fillWeightsRandom();
	}

	public void fillRandomBiases() {
		foreach (Layer layer in layers)
			layer.fillBiasesRandom();
	}

	public void putData(EList<double> data) {
		for (int r = 0; r < layers.First().input.rows; r++)
		for (int c = 0; c < layers.First().input.columns; c++)
			layers.First().input[r, c].value = data[r, c];
	}

	public void setInputLength(int length) {
		if (layers.Count > 0) layers[0] = new SimpleLayer(length);
		else addSimpleLayer(length);
		setUnitIdsForLastLayer();
	}

	public void addSimpleLayer(int length) {
		SimpleLayer layer = layers.Count == 0 ? new SimpleLayer(length) : new SimpleLayer(layers.Last());

		layers.Add(layer);
		setUnitIdsForLastLayer();
	}

	public void addDenceLayer(int length) {
		if (layers.Count == 0)
			throw new Exception("Dence layer can not be first one");

		layers.Add(new DenceLayer(length, layers.Last()));
		setUnitIdsForLastLayer();
	}

	public void addConvolutionalLayer(Filter filter, int filtersAmount, int stride) {
		if (layers.Count == 0)
			throw new Exception("Convolutional layer can not be first one");

		layers.Add(new ConvolutionalLayer(layers.Last(), filter, filtersAmount, stride));
		setUnitIdsForLastLayer();
	}

	public void addPoolingLayer(Filter filter, int stride, PoolingNode.PoolingMethod method) {
		if (layers.Count == 0)
			throw new Exception("Pooling layer can not be first one");

		layers.Add(new PoolingLayer(layers.Last(), filter, stride, method));
		setUnitIdsForLastLayer();
	}

	public void backpropagate(EList<double> expectedOutput, double learningFactor) {
		layers.Last().countDerivatives(expectedOutput);

		for (int i = layers.Count - 1; i > 1; i--)
			layers[i].countDerivatives();

		for (int i = layers.Count - 1; i > 1; i--)
			layers[i].applyDerivativesToWeights(learningFactor);

		if (!useBiases) return;
		for (int i = layers.Count - 1; i > 1; i--)
			layers[i].applyDerivativesToBiases(learningFactor);
	}

	public string serialize() {
		JObject nn = new JObject {
			["name"] = name, 
			["description"] = description
		};

		JArray layersArray = new JArray();
		foreach (Layer layer in layers) layersArray.Add(layer.toJObject());
		nn["layers"] = layersArray;

		return nn.ToString();
	}

	public static NeuralNetwork deserialize(string str) => new NeuralNetwork();

	private void setUnitIdsForLastLayer() {
		Layer layer = layers.Last();
		int layerIndex = layers.Count - 1;

		// TODO: Change to IEnumerator instead of .input
		for (int i = 0; i < layer.input.Count; i++) layer.input[i].id = $"{layerIndex}_{i}";
	}
}

}