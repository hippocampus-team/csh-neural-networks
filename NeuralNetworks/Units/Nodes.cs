using System.Collections.Generic;
using NeuralNetworks.Misc;

namespace NeuralNetworks.Units {

public class Node : Unit {
	public override double inactivatedValue { get => value; protected set => this.value = value; }
	
	public Node() : this(0) { }
	public Node(double value) {
		this.value = value;
		derivative = 1;
	}
	
	public override void count() { }
	public override void countDerivativesOfInputUnits() { }
	public static Node getEmpty() => new Node();
}

public class ReferNode : Node {
	public Unit inputUnit { get; protected set; }
	
	protected ReferNode() { }
	public ReferNode(Unit inputUnit) {
		this.inputUnit = inputUnit;
		value = inputUnit.value;
		derivative = inputUnit.derivative;
	}

	public override void count() => value = inputUnit.value;
	public override void countDerivativesOfInputUnits() => inputUnit.derivative = derivative;
	public new static ReferNode getEmpty() => new ReferNode();
}

public class PoolingNode : Node {
	public List<Unit> inputUnits { get; protected set; }
	public PoolingMethod poolingMethod { get; protected set; }

	protected PoolingNode() { }
	public PoolingNode(List<Unit> inputUnits, PoolingMethod poolingMethod) {
		value = 0;
		derivative = 0;
		this.poolingMethod = poolingMethod;
		this.inputUnits = inputUnits;
	}

	public override void count() => value = poolingMethod.count(inputUnits);
	public override void countDerivativesOfInputUnits() => poolingMethod.countDerivativesOfInputUnits(inputUnits, derivative);
	
	public new static PoolingNode getEmpty() => new PoolingNode();
}

}