using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Misc;
using NeuralNetworks.Units;
using Newtonsoft.Json.Linq;

namespace NeuralNetworks.Layers {

public abstract class Layer {
	public abstract EList<Unit> input { get; protected set; }
	public abstract EList<Unit> output { get; protected set; }
	public abstract IEnumerable<Unit> neurons { get; }

	public abstract void count();

	public abstract void fillWeightsRandom();
	public abstract void fillBiasesRandom();

	public abstract void countDerivatives();
	public abstract void countDerivatives(EList<double> expectedOutput);

	public abstract void applyDerivativesToWeights(double learningFactor);
	public abstract void applyDerivativesToBiases(double learningFactor);

	public abstract EList<double> getInputValues();
	public abstract EList<double> getOutputValues();
	
	public virtual Unit this[int neuronIndex] => input[neuronIndex];
	
	public virtual JObject toJObject() {
		JObject layer = new JObject {["type"] = GetType().Name};

		JArray unitsArray = new JArray();
		foreach (Unit unit in neurons) unitsArray.Add(unit.toJObject());
		layer["units"] = unitsArray;

		return layer;
	}
	public virtual Layer fillFromJObject(JObject json) {
		JArray unitsJArray = json["units"]!.Value<JArray>();

		EList<Unit> units = 
			new EList<Unit>((from JObject unitJson in unitsJArray 
							 let typeName = unitJson["type"]!.Value<string>() 
							 let unitType = Type.GetType("NeuralNetworks.Units." + typeName) 
							 let initUnitMethod = unitType!.GetMethod("fillFromJObject") 
							 let unit = (Unit) unitType.GetMethod("getEmpty")?.Invoke(null, null) 
							 select unit.fillFromJObject(unitJson)).ToList());

		input = units;
		output = units;
		return this;
	}
}

}