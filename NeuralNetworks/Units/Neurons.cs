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
			bias = Constants.defaultBias;

			for (int i = 0; i < inputUnits.Count; i++)
				weights.Add(1);
		}

		public override void Count() {
			double weightedSum = inputUnits.Select((unit, i) => unit.value * weights[i]).Sum();
			value = MathTools.Sigmoid(weightedSum + bias);
		}

		public override void CountDerivatives() {
			for (int i = 0; i < inputUnits.Count; i++)
				inputUnits[i].derivative += derivative * MathTools.SigmoidDerivative(value) * weights[i];
		}

		public override void ApplyDerivativesToWeights(double learningFactor) {
			for (int i = 0; i < weights.Count; i++)
				weights[i] += derivative * inputUnits[i].value * MathTools.SigmoidDerivative(value) * learningFactor;
		}

		public override void ApplyDerivativesToBias(double learningFactor) =>
			bias += derivative * MathTools.SigmoidDerivative(value) * learningFactor;
		
		public override JObject ToJObject() {
			JObject unit = base.ToJObject();
			
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

		public ConvolutionalNeuron(EList<Unit> inputUnits, Filter filter, List<int> indexes, int column)
			: base(inputUnits) {
			this.filter = filter;
			this.indexes = indexes;
			this.column = column;
		}

		public override void Count() {
			double weightedSum = 0;

			for (int r = 0; r < inputUnits.rows; r++)
				weightedSum += inputUnits[r, column].value * filter.values[indexes[r]];

			value = MathTools.Sigmoid(weightedSum + bias);
		}

		public override void CountDerivatives() {
			for (int r = 0; r < inputUnits.rows; r++)
				inputUnits[r, column].derivative +=
					derivative * MathTools.SigmoidDerivative(value) * filter.values[indexes[r]];
		}

		public override void ApplyDerivativesToWeights(double learningFactor) {
			for (int w = 0; w < weights.Count; w++)
				filter.values[indexes[w]] += learningFactor * derivative
					* inputUnits[w, column].value * MathTools.SigmoidDerivative(value);
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