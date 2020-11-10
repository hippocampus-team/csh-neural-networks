using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Units;

namespace NeuralNetworks.Misc {

public interface PoolingMethod {
	double count(List<Unit> inputUnits);
	void countDerivativesOfInputUnits(List<Unit> inputUnits, double derivative);
}

public class AveragePooling : PoolingMethod {
	public double count(List<Unit> inputUnits) {
		double sum = inputUnits.Sum(unit => unit.value);
		return sum / inputUnits.Count;
	}

	public void countDerivativesOfInputUnits(List<Unit> inputUnits, double derivative) {
		double inputDerivative = derivative / inputUnits.Count;
		foreach (Unit unit in inputUnits) unit.derivative = inputDerivative;
	}
}

public class MaxPooling : PoolingMethod {
	private int lastMaxIndex;
	
	public double count(List<Unit> inputUnits) {
		double max = inputUnits[0].value;
		lastMaxIndex = 0;

		for (int i = 1; i < inputUnits.Count; i++) {
			if (!(inputUnits[i].value > max)) continue;
			max = inputUnits[i].value;
			lastMaxIndex = i;
		}

		return max;
	}

	public void countDerivativesOfInputUnits(List<Unit> inputUnits, double derivative) {
		foreach (Unit unit in inputUnits) unit.derivative = 0;
		inputUnits[lastMaxIndex].derivative = derivative;
	}
}

public class MinPooling : PoolingMethod {
	private int lastMinIndex;
	
	public double count(List<Unit> inputUnits) {
		double min = inputUnits[0].value;
		lastMinIndex = 0;

		for (int i = 1; i < inputUnits.Count; i++) {
			if (inputUnits[i].value < min) {
				min = inputUnits[i].value;
				lastMinIndex = i;
			}
		}

		return min;
	}

	public void countDerivativesOfInputUnits(List<Unit> inputUnits, double derivative) {
		foreach (Unit unit in inputUnits) unit.derivative = 0;
		inputUnits[lastMinIndex].derivative = derivative;
	}
}

}