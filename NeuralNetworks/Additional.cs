using System;

namespace NeuralNetworks {
public static class Tools {
	public static double Sigmoid(double value) {
		float k = (float) Math.Exp(value);
		return k / (1.0f + k);
	}

	public static double SigmoidDerivative(double value) => value * (1 - value);

	public static double Clamp(double val, double min, double max) {
		if (val.CompareTo(min) < 0) return min;
		if (val.CompareTo(max) > 0) return max;
		return val;
	}
}

public static class Constants {
	public const double weightRandomFillSpread = 4;

	public const double weightMutatedMax = 5;
	public const double weightMutatedMin = -5;
	public const double weightMutationSpread = 4;


	public const double defaultBias = 0;

	public const double biasRandomFillSpread = 1;

	public const double biasMutatedMax = 3;
	public const double biasMutatedMin = -3;
	public const double biasMutationSpread = 1;
}
}