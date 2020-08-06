using System;

namespace NeuralNetworks {
	
	public static class MathTools {
		public static double Sigmoid(double value) {
			double k = Math.Exp(value);
			double d = k / (1.0f + k);
			return d;
		}

		public static double SigmoidDerivative(double value) => value * (1 - value);

		public static double Clamp(double val, double min, double max) {
			if (val.CompareTo(min) < 0) return min;
			if (val.CompareTo(max) > 0) return max;
			return val;
		}
	}
}