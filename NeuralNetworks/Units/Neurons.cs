using System;
using System.Linq;
using System.Collections.Generic;
using NeuralNetworks.Misc;
using Newtonsoft.Json.Linq;

namespace NeuralNetworks.Units {

public class Neuron : Unit {
	public List<Unit> inputUnits { get; protected set; }
	public List<double> weights { get; private set; }
	public double bias { get; set; }
	public ActivationFunction activationFunction { get; set; }
	public override double inactivatedValue { get; protected set; }

	protected Neuron() { }
	public Neuron(IEnumerable<Unit> inputUnits, ActivationFunction activationFunction) {
		value = 0;
		weights = new List<double>();
		this.inputUnits = inputUnits.ToList();
		bias = Constants.defaultBias;
		this.activationFunction = activationFunction;

		foreach (Unit inputUnit in this.inputUnits) weights.Add(1);
	}

	public override void count() {
		double weightedSum = 0;
		for (int i = 0; i < inputUnits.Count; i++)
			weightedSum += inputUnits[i].activatedValue * weights[i];
		value = activationFunction.count(weightedSum + bias);
		
		// Reset derivative after each iteration
		derivative = 0;
	}
	
	// Derivative var in neuron is a derivative of inactivated value in relation to the cost function

	// Actually counts derivatives of units in previous layer
	public override void countDerivativesOfInputUnits() {
		for (int i = 0; i < inputUnits.Count; i++)
			inputUnits[i].derivative += derivative * weights[i] * activationFunction.countDerivative(inputUnits[i].inactivatedValue);
	}
	
	public override void countDerivative(double expectedOutput) {
		derivative = 2 * (value - expectedOutput) * activationFunction.countDerivative(inactivatedValue);
	}

	public virtual void applyDerivativeToWeights(double learningFactor) {
		for (int i = 0; i < weights.Count; i++)
			weights[i] += derivative * inputUnits[i].activatedValue * learningFactor;
	}

	public virtual void applyDerivativeToBias(double learningFactor) =>
		bias += derivative * learningFactor;

	public override JObject toJObject() {
		JObject unit = base.toJObject();
		
		JArray inputUnitsIds = new JArray();
		foreach (Unit inputUnit in inputUnits) inputUnitsIds.Add(inputUnit.id);
		unit["inputs"] = inputUnitsIds;

		unit["bias"] = bias;

		JArray weightsArray = new JArray();
		foreach (double weight in weights) weightsArray.Add(weight);
		unit["weights"] = weightsArray;

		unit["af"] = activationFunction.GetType().Name;

		return unit;
	}
	
	public override Unit fillFromJObject(JObject json) {
		base.fillFromJObject(json);
		
		inputUnits = new List<Unit>();
		
		JArray inputUnitsJArray = json["inputs"]!.Value<JArray>();
		foreach (JToken unitIdJson in inputUnitsJArray)
			inputUnits.Add(ConstructionNeuronIndexer.activeIndexer.getUnitById(unitIdJson.Value<string>()));

		bias = json["bias"]!.Value<double>();

		weights = new List<double>();
		JArray weightsJArray = json["weights"]!.Value<JArray>();
		foreach (JToken weightJson in weightsJArray)
			weights.Add(weightJson.Value<double>());

		activationFunction = (ActivationFunction) 
			Activator.CreateInstance(Type.GetType("NeuralNetworks." + json["af"]!.Value<string>())!);

		return this;
	}
	
	public static Neuron getEmpty() => new Neuron();
}

public class ConvolutionalNeuron : Neuron {
	private Filter filter { get; }
	private List<int> indexes { get; }
	
	private ConvolutionalNeuron() { }
	public ConvolutionalNeuron(List<Unit> inputUnits, Filter filter, List<int> indexes, 
							   ActivationFunction activationFunction) :
		base(inputUnits, activationFunction) {
		this.filter = filter;
		this.indexes = indexes;
	}

	public override void count() {
		double weightedSum = 0;
		
		for (int i = 0; i < filter.values.Count; i++)
			weightedSum += inputUnits[indexes[i]].value * filter.values[i];

		value = activationFunction.count(weightedSum + bias);
	}

	public override void countDerivativesOfInputUnits() {
		for (int i = 0; i < filter.values.Count; i++) {
			inputUnits[indexes[i]].derivative +=
				derivative * activationFunction.countDerivative(value) * filter.values[i];
		}
	}

	public override void applyDerivativeToWeights(double learningFactor) {
		for (int i = 0; i < filter.values.Count; i++) {
			filter.values[i] += learningFactor * derivative * inputUnits[indexes[i]].value 
								* activationFunction.countDerivative(value);
		}
	}
	
	public new static ConvolutionalNeuron getEmpty() => new ConvolutionalNeuron();
}

public interface TransformableToNormalNeuron {
	public Neuron toNormalNeuron();
}

}