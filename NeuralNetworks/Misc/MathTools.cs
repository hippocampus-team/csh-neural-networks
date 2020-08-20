using System;

namespace NeuralNetworks.Misc {

public static class MathTools {
	public static double sigmoid(double value) {
		if (value < -50.0) return 0.0;
		if (value > 50.0) return 1.0;

		double k = Math.Exp(value);
		return k / (1.0d + k);
	}

	public static double sigmoidDerivative(double value) => value * (1 - value);

	public static double clamp(double val, double min, double max) {
		if (val.CompareTo(min) < 0) return min;
		if (val.CompareTo(max) > 0) return max;
		return val;
	}
}

}