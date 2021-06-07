using System;
using System.Collections.Generic;
using NeuralNetworks.Misc;
using NeuralNetworks.Units;
using Newtonsoft.Json.Linq;

namespace NeuralNetworks.Layers {

public class DenseLayer : SameInputOutputLayer {
	public override IEnumerable<Unit> units => neurons;
	public override LayerType layerType => LayerType.dense;

	private readonly List<Neuron> neurons;

	private DenseLayer() {
		neurons = new List<Neuron>();
		input = new DenseLayerConnection(neurons);
	}

	public DenseLayer(LayerConnection inputConnection, int length, ActivationFunction activationFunction) : this() {
		for (int i = 0; i < length; i++)
			neurons.Add(new Neuron(inputConnection.enumerable, activationFunction));
	}

	public override void count() {
		foreach (Neuron neuron in neurons)
			neuron.count();
	}

	public override void fillParametersRandomly() {
		Random rnd = new Random(Guid.NewGuid().GetHashCode());

		foreach (Neuron neuron in neurons)
			for (int i = 0; i < neuron.weights.Count; i++)
				neuron.weights[i] = (rnd.NextDouble() - 0.5) * Constants.weightRandomFillSpread;
		
		foreach (Neuron neuron in neurons)
			neuron.bias = (rnd.NextDouble() - 0.5) * Constants.biasRandomFillSpread;
	}

	public override void countDerivativesOfPreviousLayer() {
		foreach (Neuron neuron in neurons)
			neuron.countDerivativesOfInputUnits();
	}

	public override void countDerivatives(List<double> expectedOutput) {
		for (int i = 0; i < neurons.Count; i++)
			neurons[i].countDerivative(expectedOutput[i]);
	}

	public override void applyDerivativesToParameters(double learningFactor) {
		foreach (Neuron neuron in neurons)
			neuron.applyDerivativeToParameters(learningFactor);
	}

	public override List<double> getInputValues() {
		List<double> values = new List<double>();

		foreach (Neuron neuron in neurons)
			values.Add(neuron.value);

		return values;
	}

	public override Layer fillFromJObject(JObject json) {
		JArray unitsJArray = json["units"]!.Value<JArray>();
		
		foreach (JToken unitToken in unitsJArray) {
			Unit unit = JFactory.constructUnit((JObject) unitToken);
			
			if (!(unit is Neuron)) throw new 
				ArgumentException("Only neurons are allowed in dense layers. Failed at " + unitToken.Path);
			
			neurons.Add((Neuron) unit);
		}
		
		return this;
	}
	
	public static DenseLayer getEmpty() => new DenseLayer();

	private class DenseLayerConnection : NoDepthLayerConnection {
		private readonly List<Neuron> neurons;
		
		public override IEnumerable<Unit> enumerable => neurons;

		public override Unit this[int index] {
			get => neurons[index];
			set => neurons[index] = (Neuron) value;
		}

		public override int length => neurons.Count;

		public DenseLayerConnection(List<Neuron> neurons) {
			this.neurons = neurons;
		}
	}
}

}