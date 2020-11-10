using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Layers;
using NeuralNetworks.Misc;
using Newtonsoft.Json.Linq;

namespace NeuralNetworks {

public class NeuralNetwork {
	public string name { get; set; }
	public string description { get; set; }
	
	public List<Layer> layers { get; private set; }
	public bool useBiases { get; }

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
		1 - Math.Pow(expectedOutput - layers[^1].output[index].value, 2);

	public IEnumerable<double> getCosts(IEnumerable<double> expectedOutput) =>
		expectedOutput.Select((value, i) => 1 - Math.Pow(value - layers[^1].output[i].value, 2)).ToList();

	public double getTotalCost(IEnumerable<double> expectedOutput) => getCosts(expectedOutput).Average();

	public int getMaxIndexInOutput() {
		List<double> outputs = layers[^1].getOutputValues();

		int index = 0;
		for (int i = 1; i < outputs.Count; i++) {
			if (outputs[i] > outputs[index])
				index = i;
		}
		return index;
	}

	public void fillPropertiesRandomly() {
		foreach (Layer layer in layers)
			layer.fillPropertiesRandomly();
	}

	public void putData(List<double> data) {
		for (int i = 0; i < layers[0].input.length; i++)
			layers[0].input[i].value = data[i];
	}

	public void setInputLength(int length) {
		if (layers.Count > 0) {
			layers[0] = new SimpleLayer(length);
			layers[0].setUnitsIds(0);
		}
		else {
			addSimpleLayer(length);
			layers[^1].setUnitsIds(layers.Count - 1);
		}
	}

	public void addSimpleLayer(int length) {
		SimpleLayer layer = layers.Count == 0 ? new SimpleLayer(length) : new SimpleLayer(layers[^1].output);

		layers.Add(layer);
		layers[^1].setUnitsIds(layers.Count - 1);
	}

	public void addDenseLayer(int length, ActivationFunction activationFunction) {
		if (layers.Count == 0)
			throw new Exception("Dense layer can not be first one");

		layers.Add(new DenseLayer(layers[^1].output, length, activationFunction));
		layers[^1].setUnitsIds(layers.Count - 1);
	}

	public void backpropagate(List<double> expectedOutput, double learningFactor) {
		layers[^1].countDerivatives(expectedOutput);

		for (int i = layers.Count - 1; i > 0; i--)
			layers[i].countDerivativesOfPreviousLayer();

		for (int i = layers.Count - 1; i >= 0; i--)
			layers[i].applyDerivativesToWeights(learningFactor);

		if (!useBiases) return;
		for (int i = layers.Count - 1; i >= 0; i--)
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

	public static NeuralNetwork deserialize(string jsonString) {
		JObject json;
		try {
			json = JObject.Parse(jsonString);
		} catch (Exception e) {
			Console.WriteLine("Error while parsing json: ");
			Console.WriteLine(e);
			throw;
		}

		NeuralNetwork nn = new NeuralNetwork {
			name = json["name"]?.Value<string>() ?? "", 
			description = json["description"]?.Value<string>() ?? "",
			layers = new List<Layer>()
		};

		ConstructionNeuronIndexer.startConstruction(nn);
		
		try {
			JArray layersJson = json["layers"]!.Value<JArray>();
			foreach (JToken layerJsonToken in layersJson) {
				JObject layerJson = (JObject) layerJsonToken;
				
				string typeName = layerJson["type"]?.Value<string>() 
								  ?? throw new ArgumentException("Layer type not found in " + layerJsonToken.Path);
				
				Type layerType = Type.GetType("NeuralNetworks.Layers." + typeName);
				if (layerType == null) throw new ArgumentException("Wrong layer type of " + layerJsonToken.Path);
				
				Layer layer = (Layer) layerType.GetMethod("getEmpty")?.Invoke(null, null);
				if (layer == null) throw new ArgumentException("Wrong layer type of " + layerJsonToken.Path);
				
				nn.layers.Add(layer.fillFromJObject(layerJson));
			}
		} catch (Exception e) {
			Console.WriteLine("Error while constructing neural network: ");
			Console.WriteLine(e);
			throw;
		}

		ConstructionNeuronIndexer.endConstruction();

		return nn;
	}
}

}