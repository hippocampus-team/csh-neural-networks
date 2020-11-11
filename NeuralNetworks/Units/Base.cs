using Newtonsoft.Json.Linq;

namespace NeuralNetworks.Units {

public abstract class Unit {
	public string id { get; set; }
	public double value { get; set; }
	public double derivative { get; set; }

	public abstract void count();
	public abstract void countDerivativesOfInputUnits();
	public virtual void countDerivative(double expectedOutput) => derivative = 2 * (value - expectedOutput);

	public virtual JObject toJObject() {
		JObject unit = new JObject {
			["id"] = id, ["type"] = GetType().Name
		};
		return unit;
	}
	
	public virtual Unit fillFromJObject(JObject json) {
		id = json["id"]!.Value<string>();
		return this;
	}
}

}