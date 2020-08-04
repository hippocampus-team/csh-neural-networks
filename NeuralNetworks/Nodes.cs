using System.Collections.Generic;
using NeuralNetworks.Misc;

namespace NeuralNetworks {

public abstract class Unit {
	public double Value { get; set; }
	public List<double> Weights { get; set; }
	public double Bias { get; set; }
	public double Derivative { get; set; }

	protected EList<Unit> InputUnits { get; set; }

	public abstract void Count();
	public abstract void CountDerivatives();
	public abstract void ApplyDerivativesToWeights(double learningFactor);
	public abstract void ApplyDerivativesToBias(double learningFactor);
}

public class Node : Unit {
	public Node() : this(0) { }

	public Node(double value) {
		Value = value;
		Weights = new List<double>();
		Bias = 0;
		Derivative = 0;

		InputUnits = new EList<Unit>();
	}

	public override void Count() { }
	public override void CountDerivatives() { }
	public override void ApplyDerivativesToWeights(double learningFactor) { }
	public override void ApplyDerivativesToBias(double learningFactor) { }
}

public class ReferNode : Node {
	public ReferNode(Unit inputUnit) {
		Value = inputUnit.Value;
		Weights = new List<double>();
		Bias = 0;
		Derivative = 0;

		Weights.Add(1);

		InputUnits = new EList<Unit> { inputUnit };
	}

	public override void Count() => Value = InputUnits[0].Value;

	public override void CountDerivatives() =>
		// Derivative = OutputUnits[0].Derivative;
		InputUnits[0].Derivative = Derivative;
}

// public class ArrayReferNode : Node {
// 	public ArrayReferNode(StackUnit stackRefer, int stackIndex) : this(stackRefer.Values, stackIndex) { }
//
// 	public ArrayReferNode(List<double> array, int index) : base(array[index]) {
// 		Weights.Add(1);
//
// 		ArrayRefer = array;
// 		ArrayIndex = index;
// 	}
//
// 	private List<double> ArrayRefer { get; }
// 	private int ArrayIndex { get; }
//
// 	public override void Count() => Value = ArrayRefer[ArrayIndex];
//
// 	public override void CountDerivatives() => Derivative = OutputUnits[0].Derivative;
// }

public class PoolingNode : Node {
	public PoolingNode(EList<Unit> inputUnits) : this(inputUnits, 0) { }

	public PoolingNode(EList<Unit> inputUnits, PoolingMethod method) {
		Value = 0;
		Weights = new List<double>();
		Bias = 0;
		Derivative = 0;

		Method = method;

		InputUnits = inputUnits;
	}

	public PoolingMethod Method { get; set; }

	public override void Count() {
		switch (Method) {
			case PoolingMethod.average:
				double average = 0;

				foreach (Unit unit in InputUnits)
					average += unit.Value;

				Value = average / InputUnits.Count;
				break;

			case PoolingMethod.max:
				double max = InputUnits[0].Value;

				for (int i = 1; i < InputUnits.Count; i++)
					if (InputUnits[i].Value > max)
						max = InputUnits[i].Value;

				Value = max;
				break;

			case PoolingMethod.min:
				double min = InputUnits[0].Value;

				for (int i = 1; i < InputUnits.Count; i++)
					if (InputUnits[i].Value < min)
						min = InputUnits[i].Value;

				Value = min;
				break;
		}
	}

	public enum PoolingMethod {
		average,
		max,
		min
	}
}

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

		Value = Tools.Sigmoid(weightedSum + Bias);
	}

	public override void CountDerivatives() {
		for (int i = 0; i < InputUnits.Count; i++)
			InputUnits[i].Derivative += Derivative * Tools.SigmoidDerivative(Value) * Weights[i];
	}

	public override void ApplyDerivativesToWeights(double learningFactor) {
		for (int i = 0; i < Weights.Count; i++)
			Weights[i] += Derivative * InputUnits[i].Value * Tools.SigmoidDerivative(Value) * learningFactor;
	}

	public override void ApplyDerivativesToBias(double learningFactor) =>
		Bias += Derivative * Tools.SigmoidDerivative(Value) * learningFactor;
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

			Value = Tools.Sigmoid(weightedSum + Bias);
		}
	}

	public override void CountDerivatives() {
		for (int c = 0; c < InputUnits.Columns; c++)
		for (int r = 0; r < InputUnits.Rows; r++)
			InputUnits[r, c].Derivative +=
				Derivative * Tools.SigmoidDerivative(Value) * Filters[c].Values[Indexes[r]];
	}

	public override void ApplyDerivativesToWeights(double learningFactor) {
		for (int w = 0; w < Weights.Count; w++)
		for (int c = 0; c < InputUnits.Columns; c++)
			Filters[c].Values[Indexes[w]] += learningFactor *
				Derivative * InputUnits[w].Value * Tools.SigmoidDerivative(Value);
	}
}

}