using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Misc;
using Newtonsoft.Json.Linq;

namespace NeuralNetworks.Units {

public class Neuron : Unit {
	public List<double> weights { get; }
	public double bias { get; set; }

	public Neuron(EList<Unit> inputUnits) {
		value = 0;
		weights = new List<double>();
		this.inputUnits = inputUnits;
		bias = Constants.DEFAULT_BIAS;

		for (int i = 0; i < inputUnits.Count; i++)
			weights.Add(1);
	}

	public override void count() {
		double weightedSum = inputUnits.Select((unit, i) => unit.value * weights[i]).Sum();
		value = MathTools.sigmoid(weightedSum + bias);
	}

	public override void countDerivatives() {
		for (int i = 0; i < inputUnits.Count; i++)
			inputUnits[i].derivative += derivative * MathTools.sigmoidDerivative(value) * weights[i];
	}

	public override void applyDerivativesToWeights(double learningFactor) {
		for (int i = 0; i < weights.Count; i++)
			weights[i] += derivative * inputUnits[i].value * MathTools.sigmoidDerivative(value) * learningFactor;
	}

	public override void applyDerivativesToBias(double learningFactor) =>
		bias += derivative * MathTools.sigmoidDerivative(value) * learningFactor;

	public override JObject toJObject() {
		JObject unit = base.toJObject();

		unit["bias"] = bias;

		JArray weightsArray = new JArray();
		foreach (double weight in weights) weightsArray.Add(weight);
		unit["weights"] = weightsArray;

		return unit;
	}
}

public class ConvolutionalNeuron : Neuron {
	private Filter filter { get; }
	private List<int> indexes { get; }
	private int column { get; }

	public ConvolutionalNeuron(EList<Unit> inputUnits, Filter filter, List<int> indexes, int column) :
		base(inputUnits) {
		this.filter = filter;
		this.indexes = indexes;
		this.column = column;
	}

	public override void count() {
		double weightedSum = 0;
		
		for (int i = 0; i < filter.values.Count; i++)
			weightedSum += inputUnits[indexes[i], column].value * filter.values[i];

		value = MathTools.sigmoid(weightedSum + bias);
	}

	public override void countDerivatives() {
		for (int i = 0; i < filter.values.Count; i++) {
			inputUnits[indexes[i], column].derivative +=
				derivative * MathTools.sigmoidDerivative(value) * filter.values[i];
		}
	}

	public override void applyDerivativesToWeights(double learningFactor) {
		for (int i = 0; i < filter.values.Count; i++) {
			filter.values[i] += learningFactor * derivative * inputUnits[indexes[i], column].value *
										 MathTools.sigmoidDerivative(value);
		}
	}
}

// public class SharedNeuron : Neuron {
// 	public SharedNeuron(List<double> sharedWeights, List<int> indexes) :
// 		this(new List<Unit>(), sharedWeights, indexes) { }
//
// 	public SharedNeuron(List<Unit> inputUnits, List<double> sharedWeights, List<int> indexes) : base(inputUnits) {
// 		SharedWeights = sharedWeights;
// 		Indexes = indexes;
//
// 		FetchWeights();
// 	}
//
// 	public List<double> SharedWeights { get; set; }
// 	public List<int> Indexes { get; set; }
//
// 	public override void Count() {
// 		double WeightedSum = 0;
//
// 		for (int i = 0; i < InputUnits.Count; i++)
// 			WeightedSum += InputUnits[i].Value * SharedWeights[Indexes[i]];
//
// 		Value = Tools.Sigmoid(WeightedSum - Bias);
// 	}
//
// 	public override void CountDerivatives() {
// 		//FetchWeights();
// 		//base.CountDerivatives();
//
// 		for (int i = 0; i < InputUnits.Count; i++)
// 			InputUnits[i].Derivative += Derivative * Tools.SigmoidDerivative(Value) * SharedWeights[Indexes[i]];
// 	}
//
// 	public override void ApplyDerivativesToWeights(double learningFactor) {
// 		for (int i = 0; i < Weights.Count; i++)
// 			SharedWeights[Indexes[i]] +=
// 				Derivative * InputUnits[i].Value * Tools.SigmoidDerivative(Value) * learningFactor;
// 	}
//
// 	private void FetchWeights() {
// 		for (int i = 0; i < Weights.Count; i++)
// 			Weights[i] = SharedWeights[Indexes[i]];
// 	}
// }

}