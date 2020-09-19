using NeuralNetworks.Misc;
using Newtonsoft.Json.Linq;

namespace NeuralNetworks.Units {

public abstract class Unit {
	public string id { get; set; }
	public double value { get; set; }
	public double derivative { get; set; }

	public EList<Unit> inputUnits { get; protected set; }

	public abstract void count();
	public abstract void countDerivatives();
	public abstract void applyDerivativesToWeights(double learningFactor);
	public abstract void applyDerivativesToBias(double learningFactor);

	public virtual JObject toJObject() {
		JObject unit = new JObject {
			["id"] = id, ["type"] = GetType().Name, ["value"] = value, ["derivative"] = derivative
		};

		JArray inputUnitsIds = new JArray();
		foreach (Unit inputUnit in inputUnits) inputUnitsIds.Add(inputUnit.id);
		unit["inputs"] = inputUnitsIds;

		return unit;
	}
	
	public virtual Unit fillFromJObject(JObject json) {
		id = json["id"]!.Value<string>();
		value = json["value"]!.Value<double>();
		derivative = json["derivative"]!.Value<double>();
		
		inputUnits = new EList<Unit>();

		JArray inputUnitsJArray = json["inputs"]!.Value<JArray>();
		foreach (JToken unitIdJson in inputUnitsJArray)
			inputUnits.Add(ConstructionNeuronIndexer.activeIndexer.getUnitById(unitIdJson.Value<string>()));

		return this;
	}
}

}