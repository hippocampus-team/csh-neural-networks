using System.Collections.Generic;
using NeuralNetworks.Misc;

namespace NeuralNetworks {

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
			case PoolingMethod.Average:
				double average = 0;

				foreach (Unit unit in InputUnits)
					average += unit.Value;

				Value = average / InputUnits.Count;
				break;

			case PoolingMethod.Max:
				double max = InputUnits[0].Value;

				for (int i = 1; i < InputUnits.Count; i++)
					if (InputUnits[i].Value > max)
						max = InputUnits[i].Value;

				Value = max;
				break;

			case PoolingMethod.Min:
				double min = InputUnits[0].Value;

				for (int i = 1; i < InputUnits.Count; i++)
					if (InputUnits[i].Value < min)
						min = InputUnits[i].Value;

				Value = min;
				break;
		}
	}

	public enum PoolingMethod {
		Average,
		Max,
		Min
	}
}

}