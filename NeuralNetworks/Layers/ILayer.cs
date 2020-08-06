using NeuralNetworks.Misc;
using NeuralNetworks.Units;

namespace NeuralNetworks.Layers {
	public interface ILayer {
		EList<Unit> input { get; }
		EList<Unit> output { get; }
		void Count();

		void FillWeightsRandom();
		void FillBiasesRandom();

		void CountDerivatives();
		void CountDerivatives(EList<double> expectedOutput);

		void ApplyDerivativesToWeights(double learningFactor);
		void ApplyDerivativesToBiases(double learningFactor);

		EList<double> GetInputValues();
		EList<double> GetOutputValues();
	}
}