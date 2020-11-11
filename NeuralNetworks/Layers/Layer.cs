using System.Collections.Generic;
using NeuralNetworks.Units;
using Newtonsoft.Json.Linq;

namespace NeuralNetworks.Layers {

public abstract class Layer {
	public abstract LayerConnection input { get; protected set; }
	public abstract LayerConnection output { get; protected set; }
	public abstract IEnumerable<Unit> units { get; }

	public abstract void count();

	public abstract void fillParametersRandomly();

	public abstract void countDerivativesOfPreviousLayer();
	public abstract void countDerivatives(List<double> expectedOutput);
	public abstract void applyDerivativesToParameters(double learningFactor);

	public abstract List<double> getInputValues();
	public abstract List<double> getOutputValues();
	
	public virtual Unit this[int neuronIndex] => input[neuronIndex];

	public abstract JObject toJObject();
	public abstract Layer fillFromJObject(JObject json);

	public void setUnitsIds(int layerIndex) {
		int counter = 0;
		
		foreach (Unit unit in units) {
			unit.id = $"{layerIndex}_{counter}";
			counter++;
		}
	}
}

public abstract class SameInputOutputLayer : Layer {
	public sealed override LayerConnection input { get; protected set; }
	public sealed override LayerConnection output { get => input; protected set => input = value; }
	public sealed override List<double> getOutputValues() => getInputValues();
	
	public override JObject toJObject() {
		JObject layer = new JObject {["type"] = GetType().Name};

		JArray unitsArray = new JArray();
		foreach (Unit unit in input.enumerable) unitsArray.Add(unit.toJObject());
		layer["units"] = unitsArray;

		return layer;
	}
}

public interface LayerConnection {
	public IEnumerable<Unit> enumerable { get; }
	
	public Unit this[int index] { get; set; }
	public Unit this[int index, int depthIndex] { get; set; }
	
	public int length { get; }
	public int depth { get; }
}

public abstract class NoDepthLayerConnection : LayerConnection {
	public abstract IEnumerable<Unit> enumerable { get; }
	
	public abstract Unit this[int index] { get; set; }
	public Unit this[int index, int depthIndex] { get => this[index]; set => this[index] = value; }
	
	public abstract int length { get; }
	public int depth => 1;
}

}