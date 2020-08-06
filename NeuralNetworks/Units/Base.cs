using System.Collections.Generic;
using NeuralNetworks.Misc;

namespace NeuralNetworks.Units {
	public abstract class Unit {
		public double value { get; set; }
		public List<double> weights { get; set; }
		public double bias { get; set; }
		public double derivative { get; set; }

		public EList<Unit> inputUnits { get; set; }

		public abstract void Count();
		public abstract void CountDerivatives();
		public abstract void ApplyDerivativesToWeights(double learningFactor);
		public abstract void ApplyDerivativesToBias(double learningFactor);
	}
}