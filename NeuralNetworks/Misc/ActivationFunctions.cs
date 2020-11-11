using System;

namespace NeuralNetworks.Misc {

public interface ActivationFunction {
	double count(double value);
	double countDerivative(double value);
}

public class Identity : ActivationFunction {
	public double count(double value) {
		return value;
	}

	public double countDerivative(double value) {
		return 1;
	}
}

public class Sigmoid : ActivationFunction {
	private const double countApproximationLimit = 50d;
	
	public double count(double value) {
		if (value < -countApproximationLimit) return 0d;
		if (value > countApproximationLimit) return 1d;
		
		// Equivalent to 1.0d / (1.0d + Math.Exp(-value));
		double k = Math.Exp(value);
		return k / (1d + k);
	}

	public double countDerivative(double value) {
		double sigmoid = count(value);
		return sigmoid * (1d - sigmoid);
	}
}

public class HyperbolicTangent : ActivationFunction {
	private const double countApproximationLimit = 50d;
	
	public double count(double value) {
		if (value < -countApproximationLimit) return -1d;
		if (value > countApproximationLimit) return 1d;

		double a = Math.Exp(value);
		double b = Math.Exp(-value);
		return (a - b)/(a + b);
	}

	public double countDerivative(double value) {
		double tanh = count(value);
		return 1d - tanh * tanh;
	}
}

public class Relu : ActivationFunction {
	public double count(double value) {
		return value > 0 ? value : 0d;
	}

	public double countDerivative(double value) {
		return value > 0 ? 1 : 0;
	}
}

public class LeakyRelu : ActivationFunction {
	private const double leakCoefficient = 0.01d;
	
	public double count(double value) {
		return value >= 0 ? value : value * leakCoefficient;
	}

	public double countDerivative(double value) {
		return value > 0 ? 1 : leakCoefficient;
	}
}

public class Sinusoid : ActivationFunction {
	public double count(double value) {
		return Math.Sin(value);
	}

	public double countDerivative(double value) {
		return Math.Cos(value);
	}
}

}