using System;
using System.Collections.Generic;
using NeuralNetworks.Misc;

namespace NeuralNetworks {

public class ConvolutionalLayer : ILayer {
	public ConvolutionalLayer(ILayer inputLayer, Filter filter, int filtersAmount, int stride) {
		Kernels = new List<List<Filter>>();
		Model = new MatrixModel(inputLayer.Output, stride);

		for (int f = 0; f < filtersAmount; f++) {
			Kernels.Add(new List<Filter>());

			for (int c = 0; c < inputLayer.Output.Columns; c++)
				Kernels[f].Add((Filter) filter.Clone());
		}

		EList<Unit> neurons = new EList<Unit>(filtersAmount);

		for (int f = 0; f < filtersAmount; f++)
		for (int o = 0; o < Model.FilterOutputsCount(filter); o++) {
			List<int> indexes = new List<int>();

			for (int x = 0; x < filter.Count(); x++) {
				int inner = x % filter.Size + x / filter.Size * Model.Size;
				int outer = o % Model.FilterLineCount(filter) + o / Model.FilterLineCount(filter) * Model.Size;

				indexes.Add(inner + outer);
			}

			neurons.Add(new ConvolutionalNeuron(inputLayer.Output, Kernels[f], indexes));
		}

		Input = neurons;
		Output = neurons;
	}
	
	public EList<Unit> Input { get; set; }
	public EList<Unit> Output { get; set; }

	private List<List<Filter>> Kernels { get; }
	private MatrixModel Model { get; }

	public void Count() {
		foreach (Unit unit in Output)
			unit.Count();
	}

	public void FillWeightsRandom() {
		Random rnd = new Random(Guid.NewGuid().GetHashCode());

		for (int i = 0; i < Kernels.Count; i++)
		for (int j = 0; j < Kernels[i].Count; j++)
		for (int k = 0; k < Kernels[i][j].Values.Count; k++)
			Kernels[i][j].Values[k] = (rnd.NextDouble() - 0.5) * Constants.weightRandomFillSpread;
	}
	public void FillBiasesRandom() {
		Random rnd = new Random(Guid.NewGuid().GetHashCode());

		foreach (Unit unit in Output)
			unit.Bias = (rnd.NextDouble() - 0.5) * Constants.biasRandomFillSpread;
	}

	public void CountDerivatives() {
		foreach (Unit unit in Output)
			unit.CountDerivatives();
	}
	public void CountDerivatives(EList<double> expectedOutput) {
		for (int r = 0; r < Output.Rows; r++)
		for (int c = 0; c < Output.Columns; c++)
			Output[r, c].Derivative = expectedOutput[r, c] - Output[r, c].Value;
	}

	public void ApplyDerivativesToWeights(double learningFactor) {
		foreach (Unit unit in Output)
			unit.ApplyDerivativesToWeights(learningFactor);
	}
	public void ApplyDerivativesToBiases(double learningFactor) {
		foreach (Unit unit in Output)
			unit.ApplyDerivativesToBias(learningFactor);
	}
	
	public EList<double> GetInputValues() => throw new NotImplementedException();
	public EList<double> GetOutputValues() => throw new NotImplementedException();
}

}