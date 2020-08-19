using NeuralNetworks.Misc;
using NeuralNetworks.Units;
using Newtonsoft.Json.Linq;

namespace NeuralNetworks.Layers {

public abstract class Layer {
	public abstract EList<Unit> input { get; }
	public abstract EList<Unit> output { get; }

	public abstract void count();

	public abstract void fillWeightsRandom();
	public abstract void fillBiasesRandom();

	public abstract void countDerivatives();
	public abstract void countDerivatives(EList<double> expectedOutput);

	public abstract void applyDerivativesToWeights(double learningFactor);
	public abstract void applyDerivativesToBiases(double learningFactor);

	public abstract EList<double> getInputValues();
	public abstract EList<double> getOutputValues();

	public JObject toJObject() {
		JObject layer = new JObject {["type"] = GetType().Name};

		JArray unitsArray = new JArray();
		foreach (Unit unit in input) unitsArray.Add(unit.toJObject());
		layer["units"] = unitsArray;

		return layer;
	}
}

}