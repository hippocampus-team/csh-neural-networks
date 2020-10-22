using System;

namespace NeuralNetworks {

public interface ActivationFunction {
	double count(double value);
	double countDerivative(double value);
}

public class Sigmoid : ActivationFunction {
	public double count(double value) {
		if (value < -50.0) return 0.0;
		if (value > 50.0) return 1.0;
		
		// Equivalent to 1.0d / (1.0d + Math.Exp(-value));
		
		double k = Math.Exp(value);
		return k / (1.0d + k);
	}

	public double countDerivative(double value) {
		double sigmoid = count(value);
		return sigmoid * (1 - sigmoid);
	}
}

public class ModifiedSigmoid : ActivationFunction {
	public double count(double value) {
		if (value < -50.0) return 0.0;
		if (value > 50.0) return 1.0;

		double k = Math.Exp(value);
		return k / (1.0d + k);
	}
	
	public double countDerivative(double value) => value * (1 - value);
}

public class HyperbolicTangent : ActivationFunction {
	public double count(double value) {
		if (value < -50.0) return 0.0;
		if (value > 50.0) return 1.0;

		double a = Math.Exp(value);
		double b = Math.Exp(-value);
		return (a - b)/(a + b);
	}

	public double countDerivative(double value) {
		double tanh = count(value);
		return 1 - tanh * tanh;
	}
}

}