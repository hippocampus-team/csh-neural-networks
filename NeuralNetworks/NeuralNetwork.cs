using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Layers;
using NeuralNetworks.Misc;
using NeuralNetworks.Units;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace NeuralNetworks {
	public class NeuralNetwork {
		private string name { get; set; }
		private string description { get; set; }
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

		public void Run() {
			foreach (Layer layer in layers)
				layer.Count();
		}

		public double GetCost(double expectedOutput, int index) =>
			1 - Math.Pow(expectedOutput - layers.Last().output[index].value, 2);

		public IEnumerable<double> GetCosts(IEnumerable<double> expectedOutput) =>
			expectedOutput.Select((value, i) =>
				                      1 - Math.Pow(value - layers.Last().output[i].value, 2)).ToList();

		public double GetTotalCost(List<double> expectedOutput) =>
			GetCosts(expectedOutput).Average();

		public int GetMaxIndexInOutput() {
			EList<double> outputs = layers.Last().GetOutputValues();

			int index = 0;
			for (int i = 1; i < outputs.Count; i++) if (outputs[i] > outputs[index]) index = i;
			return index;
		}

		public void FillRandomWeights() {
			foreach (Layer layer in layers)
				layer.FillWeightsRandom();
		}

		public void FillRandomBiases() {
			foreach (Layer layer in layers)
				layer.FillBiasesRandom();
		}

		public void PutData(EList<double> data) {
			for (int r = 0; r < layers.First().input.rows; r++)
			for (int c = 0; c < layers.First().input.columns; c++)
				layers.First().input[r, c].value = data[r, c];
		}

		public void SetInputLength(int length) {
			if (layers.Count > 0) layers[0] = new SimpleLayer(length);
			else AddSimpleLayer(length);
			SetUnitIdsForLastLayer();
		}

		public void AddSimpleLayer(int length) {
			SimpleLayer layer = layers.Count == 0 ? new SimpleLayer(length) : new SimpleLayer(layers.Last());

			layers.Add(layer);
			SetUnitIdsForLastLayer();
		}

		public void AddDenceLayer(int length) {
			if (layers.Count == 0)
				throw new Exception("Dence layer can not be first one");

			layers.Add(new DenceLayer(length, layers.Last()));
			SetUnitIdsForLastLayer();
		}

		public void AddConvolutionalLayer(Filter filter, int filtersAmount, int stride) {
			if (layers.Count == 0)
				throw new Exception("Convolutional layer can not be first one");

			layers.Add(new ConvolutionalLayer(layers.Last(), filter, filtersAmount, stride));
			SetUnitIdsForLastLayer();
		}

		public void AddPoolingLayer(Filter filter, int stride, PoolingNode.PoolingMethod method) {
			if (layers.Count == 0)
				throw new Exception("Pooling layer can not be first one");

			layers.Add(new PoolingLayer(layers.Last(), filter, stride, method));
			SetUnitIdsForLastLayer();
		}

		public void Backpropagate(EList<double> expectedOutput, double learningFactor) {
			layers.Last().CountDerivatives(expectedOutput);

			for (int i = layers.Count - 1; i > 1; i--)
				layers[i].CountDerivatives();

			for (int i = layers.Count - 1; i > 1; i--)
				layers[i].ApplyDerivativesToWeights(learningFactor);

			if (!useBiases) return;
			for (int i = layers.Count - 1; i > 1; i--)
				layers[i].ApplyDerivativesToBiases(learningFactor);
		}

		public string Serialize() {
			JObject nn = new JObject {
				["name"] = name, 
				["description"] = description
			};

			JArray layersArray = new JArray();
			foreach (Layer layer in layers) layersArray.Add(layer.ToJObject());
			nn["layers"] = layersArray;

			return nn.ToString();
		}

		public static NeuralNetwork Deserialize(string str) {
			return new NeuralNetwork();
		}

		private void SetUnitIdsForLastLayer() {
			Layer layer = layers.Last();
			int layerIndex = layers.Count - 1;

			// TODO: Change to IEnumerator instead of .input
			for (int i = 0; i < layer.input.Count; i++) {
				layer.input[i].id = $"{layerIndex}_{i}";
			}
		}
	}
}