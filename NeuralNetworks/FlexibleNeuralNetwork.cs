using System;
using NeuralNetworks.Units;

namespace NeuralNetworks {

public class FlexibleNeuralNetwork : NeuralNetwork {
	public PrunableNeuron makeNeuronPrunable(int layerIndex, int neuronIndex) {
		Unit unit = this[layerIndex][neuronIndex];
		if (!(unit is Neuron)) throw 
			new ArgumentException($"Unit {neuronIndex} in layer {layerIndex} is not a Neuron " +
								  $"so it cannot be transformed into Prunable Neuron");
		
		PrunableNeuron prunableNeuron = new PrunableNeuron(unit as Neuron);
		this[layerIndex].input[neuronIndex] = prunableNeuron;
		return prunableNeuron;
	}
	
	public Neuron makeNeuronNormal(int layerIndex, int neuronIndex) {
		Unit unit = this[layerIndex][neuronIndex];
		if (!(unit is TransformableToNormalNeuron)) throw 
			new ArgumentException($"Unit {neuronIndex} in layer {layerIndex} is not a TransformableToNormalNeuron " +
								  $"so it cannot be transformed into Neuron");
		
		Neuron neuron = (unit as TransformableToNormalNeuron).toNormalNeuron();
		this[layerIndex].input[neuronIndex] = neuron;
		return neuron;
	}
}

}