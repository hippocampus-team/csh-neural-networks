using System.Linq;
using NeuralNetworks.Layers;
using NeuralNetworks.Misc;

namespace NeuralNetworks.Units {

public class Node : Unit {
	public Node() : this(0) { }

	public Node(double value) {
		this.value = value;
		derivative = 0;
		inputUnits = new EList<Unit>();
	}

	public override void count() { }
	public override void countDerivatives() { }
	public override void applyDerivativesToWeights(double learningFactor) { }
	public override void applyDerivativesToBias(double learningFactor) { }
	public static Node getEmpty() => new Node(0);
}

public class ReferNode : Node {
	private ReferNode() { }

	public ReferNode(Unit inputUnit) {
		value = inputUnit.value;
		derivative = 0;
		inputUnits = new EList<Unit> {inputUnit};
	}

	public override void count() => value = inputUnits[0].value;

	public override void countDerivatives() => inputUnits[0].derivative = derivative;
	
	public new static ReferNode getEmpty() => new ReferNode();
}

public class PoolingNode : Node {
	public enum PoolingMethod {
		average,
		max,
		min
	}

	private PoolingMethod method { get; }

	private PoolingNode() { }

	public PoolingNode(EList<Unit> inputUnits) : this(inputUnits, 0) { }

	public PoolingNode(EList<Unit> inputUnits, PoolingMethod method) {
		value = 0;
		derivative = 0;

		this.method = method;

		this.inputUnits = inputUnits;
	}

	public override void count() {
		switch (method) {
			case PoolingMethod.average:
				double average = inputUnits.Sum(unit => unit.value);

				value = average / inputUnits.Count;
				break;

			case PoolingMethod.max:
				double max = inputUnits[0].value;

				for (int i = 1; i < inputUnits.Count; i++) {
					if (inputUnits[i].value > max)
						max = inputUnits[i].value;
				}

				value = max;
				break;

			case PoolingMethod.min:
				double min = inputUnits[0].value;

				for (int i = 1; i < inputUnits.Count; i++) {
					if (inputUnits[i].value < min)
						min = inputUnits[i].value;
				}

				value = min;
				break;
		}
	}
	
	public new static PoolingNode getEmpty() => new PoolingNode();
}

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