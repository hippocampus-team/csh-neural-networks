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

}