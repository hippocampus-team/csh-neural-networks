using NeuralNetworks.Misc;
using NeuralNetworks.Units;
using Newtonsoft.Json.Linq;

namespace NeuralNetworks.Layers {
	public abstract class Layer {
		public abstract EList<Unit> input { get; }
		public abstract EList<Unit> output { get; }
		
		public abstract void Count();
		
		public abstract void FillWeightsRandom();
		public abstract void FillBiasesRandom();
		
		public abstract void CountDerivatives();
		public abstract void CountDerivatives(EList<double> expectedOutput);
		
		public abstract void ApplyDerivativesToWeights(double learningFactor);
		public abstract void ApplyDerivativesToBiases(double learningFactor);
		
		public abstract EList<double> GetInputValues();
		public abstract EList<double> GetOutputValues();

		public JObject ToJObject() {
			JObject layer = new JObject {
				["type"] = GetType().Name
			};

			JArray unitsArray = new JArray();
			foreach (Unit unit in input) unitsArray.Add(unit.ToJObject());
			layer["units"] = unitsArray;

			return layer;
		}
	}
}