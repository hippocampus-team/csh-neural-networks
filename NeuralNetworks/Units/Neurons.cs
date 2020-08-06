using System;
using System.Collections.Generic;
using NeuralNetworks.Misc;

namespace NeuralNetworks {

public class Neuron : Unit {
	public Neuron(EList<Unit> inputUnits) {
		Value = 0;
		Weights = new List<double>();
		InputUnits = inputUnits;
		Bias = Constants.defaultBias;

		for (int i = 0; i < inputUnits.Count; i++)
			Weights.Add(1);
	}

	public override void Count() {
		double weightedSum = 0;

		for (int i = 0; i < InputUnits.Count; i++)
			weightedSum += InputUnits[i].Value * Weights[i];

		Value = MathTools.Sigmoid(weightedSum + Bias);
	}

	public override void CountDerivatives() {
		for (int i = 0; i < InputUnits.Count; i++)
			InputUnits[i].Derivative += Derivative * MathTools.SigmoidDerivative(Value) * Weights[i];
	}

	public override void ApplyDerivativesToWeights(double learningFactor) {
		for (int i = 0; i < Weights.Count; i++)
			Weights[i] += Derivative * InputUnits[i].Value * MathTools.SigmoidDerivative(Value) * learningFactor;
	}

	public override void ApplyDerivativesToBias(double learningFactor) =>
		Bias += Derivative * MathTools.SigmoidDerivative(Value) * learningFactor;
}

public class ConvolutionalNeuron : Neuron {
	public ConvolutionalNeuron(EList<Unit> inputUnits, List<Filter> filters, List<int> indexes) 
		: base(inputUnits) {
		Filters = filters;
		Indexes = indexes;
	}

	private List<Filter> Filters { get; }
	private List<int> Indexes { get; }

	public override void Count() {
		for (int c = 0; c < InputUnits.Columns; c++) {
			double weightedSum = 0;

			for (int r = 0; r < InputUnits.Rows; r++)
				weightedSum += InputUnits[r, c].Value * Filters[c].Values[Indexes[r]];

			Value = MathTools.Sigmoid(weightedSum + Bias);
		}
	}

	public override void CountDerivatives() {
		for (int c = 0; c < InputUnits.Columns; c++)
		for (int r = 0; r < InputUnits.Rows; r++)
			InputUnits[r, c].Derivative +=
				Derivative * MathTools.SigmoidDerivative(Value) * Filters[c].Values[Indexes[r]];
	}

	public override void ApplyDerivativesToWeights(double learningFactor) {
		for (int w = 0; w < Weights.Count; w++)
		for (int c = 0; c < InputUnits.Columns; c++)
			Filters[c].Values[Indexes[w]] += learningFactor *
				Derivative * InputUnits[w].Value * MathTools.SigmoidDerivative(Value);
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