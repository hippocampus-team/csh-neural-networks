using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Misc;

namespace NeuralNetworks.Units {
	public class Node : Unit {
		public Node() : this(0) { }

		public Node(double value) {
			this.value = value;
			derivative = 0;
			inputUnits = new EList<Unit>();
		}

		public override void Count()                                          { }
		public override void CountDerivatives()                               { }
		public override void ApplyDerivativesToWeights(double learningFactor) { }
		public override void ApplyDerivativesToBias(double learningFactor)    { }
	}

	public class ReferNode : Node {
		public ReferNode(Unit inputUnit) {
			value = inputUnit.value;
			derivative = 0;
			inputUnits = new EList<Unit> {inputUnit};
		}

		public override void Count() => value = inputUnits[0].value;

		public override void CountDerivatives() =>
			inputUnits[0].derivative = derivative;
	}

	public class PoolingNode : Node {
		public enum PoolingMethod {
			Average,
			Max,
			Min
		}

		private PoolingMethod method { get; }

		public PoolingNode(EList<Unit> inputUnits) : this(inputUnits, 0) { }

		public PoolingNode(EList<Unit> inputUnits, PoolingMethod method) {
			value = 0;
			derivative = 0;

			this.method = method;

			this.inputUnits = inputUnits;
		}

		public override void Count() {
			switch (method) {
				case PoolingMethod.Average:
					double average = inputUnits.Sum(unit => unit.value);

					value = average / inputUnits.Count;
					break;

				case PoolingMethod.Max:
					double max = inputUnits[0].value;

					for (int i = 1; i < inputUnits.Count; i++)
						if (inputUnits[i].value > max)
							max = inputUnits[i].value;

					value = max;
					break;

				case PoolingMethod.Min:
					double min = inputUnits[0].value;

					for (int i = 1; i < inputUnits.Count; i++)
						if (inputUnits[i].value < min)
							min = inputUnits[i].value;

					value = min;
					break;
			}
		}
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