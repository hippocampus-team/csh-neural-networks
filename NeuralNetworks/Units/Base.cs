using NeuralNetworks.Misc;
using Newtonsoft.Json.Linq;

namespace NeuralNetworks.Units {
	public abstract class Unit {
		public string id { get; set; }
		public double value { get; set; }
		public double derivative { get; set; }

		public EList<Unit> inputUnits { get; protected set; }

		public abstract void Count();
		public abstract void CountDerivatives();
		public abstract void ApplyDerivativesToWeights(double learningFactor);
		public abstract void ApplyDerivativesToBias(double learningFactor);

		public virtual JObject ToJObject() {
			JObject unit = new JObject {
				["id"] = id,
				["type"] = GetType().Name,
				["value"] = value,
				["derivative"] = derivative
			};
			
			JArray inputUnitsIds = new JArray();
			foreach (Unit inputUnit in inputUnits) inputUnitsIds.Add(inputUnit.id);
			unit["inputs"] = inputUnitsIds;

			return unit;
		}
	}
}