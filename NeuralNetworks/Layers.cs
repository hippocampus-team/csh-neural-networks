using System;
using System.Collections.Generic;
using NeuralNetworks.Misc;

namespace NeuralNetworks {

public interface ILayer {
	void Count();

	void FillWeightsRandom();
	void FillBiasesRandom();

	void CountDerivatives();
	void CountDerivatives(EList<double> expectedOutput);

	void ApplyDerivativesToWeights(double learningFactor);
	void ApplyDerivativesToBiases(double learningFactor);

	EList<Unit> Input { get; set; }
	EList<Unit> Output { get; set; }

	EList<double> GetInputValues();
	EList<double> GetOutputValues();
}

public class DenceLayer : ILayer {
	public EList<Unit> Input { get; set; }
	public EList<Unit> Output { get; set; }
	
	public DenceLayer(int n, ILayer inputLayer) {
		EList<Unit> neurons = new EList<Unit>();

		for (int i = 0; i < n; i++)
			neurons.Add(new Neuron(inputLayer.Output));

		Input = neurons;
		Output = neurons;
	}

	//public void MutateRandomWeights()
	//{
	//    Random rnd = new Random(Guid.NewGuid().GetHashCode());

	//    int n = rnd.Next(Output.Count);
	//    Neuron neuron = (Neuron)Output[n];

	//    int w = rnd.Next(neuron.Weights.Count);
	//    neuron.Weights[w] += (rnd.NextDouble() - 0.5) * Constants.weightMutationSpread;

	//    neuron.Weights[w] = Tools.Clamp(neuron.Weights[w], Constants.weightMutatedMin, Constants.weightMutatedMax);
	//}

	//public void MutateRandomBiases()
	//{
	//    Random rnd = new Random(Guid.NewGuid().GetHashCode());

	//    int n = rnd.Next(Output.Count);
	//    Neuron neuron = (Neuron)Output[n];

	//    neuron.Bias += (rnd.NextDouble() - 0.5) * Constants.biasMutationSpread;

	//    neuron.Bias = Tools.Clamp(neuron.Bias, Constants.biasMutatedMin, Constants.biasMutatedMax);
	//}

	public void Count() {
		foreach (Unit unit in Output)
			unit.Count();
	}

	public void FillWeightsRandom() {
		Random rnd = new Random(Guid.NewGuid().GetHashCode());

		foreach (Unit unit in Output) {
			Neuron neuron = (Neuron) unit;
			for (int i = 0; i < neuron.Weights.Count; i++)
				neuron.Weights[i] = (rnd.NextDouble() - 0.5) * Constants.weightRandomFillSpread;
		}
	}

	public void FillBiasesRandom() {
		Random rnd = new Random(Guid.NewGuid().GetHashCode());

		foreach (Unit unit in Output) {
			Neuron neuron = (Neuron) unit;
			neuron.Bias = (rnd.NextDouble() - 0.5) * Constants.biasRandomFillSpread;
		}
	}

	public void CountDerivatives() {
		foreach (Unit unit in Output)
			unit.CountDerivatives();
	}

	public void CountDerivatives(EList<double> expectedOutput) {
		for (int i = 0; i < Output.Count; i++)
			Output[i].Derivative = expectedOutput[i] - Output[i].Value;
	}

	public void ApplyDerivativesToWeights(double learningFactor) {
		foreach (Unit unit in Output)
			unit.ApplyDerivativesToWeights(learningFactor);
	}

	public void ApplyDerivativesToBiases(double learningFactor) {
		foreach (Unit unit in Output)
			unit.ApplyDerivativesToBias(learningFactor);
	}
	
	public EList<double> GetInputValues() {
		EList<double> values = new EList<double>();

		foreach (Unit unit in Input)
			values.Add(unit.Value);

		return values;
	}
	public EList<double> GetOutputValues() {
		EList<double> values = new EList<double>();

		foreach (Unit unit in Output)
			values.Add(unit.Value);

		return values;
	}
}

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

public class SimpleLayer : ILayer {
	public EList<Unit> Input { get; set; }
	public EList<Unit> Output { get; set; }
	
	public SimpleLayer(int n) {
		EList<Unit> nodes = new EList<Unit>();

		for (int i = 0; i < n; i++)
			nodes.Add(new Node());

		Input = nodes;
		Output = nodes;
	}
	
	public SimpleLayer(List<double> values) {
		EList<Unit> nodes = new EList<Unit>();

		foreach (double value in values)
			nodes.Add(new Node(value));

		Input = nodes;
		Output = nodes;
	}

	public SimpleLayer(ILayer inputLayer) {
		EList<Unit> nodes = new EList<Unit>();

		foreach (Unit unit in inputLayer.Output)
			nodes.Add(new ReferNode(unit));

		Input = nodes;
		Output = nodes;
	}

	public void Count() {
		foreach (Unit unit in Output) 
			((Node) unit).Count();
	}

	public void CountDerivatives() {
		foreach (Unit unit in Input)
			unit.CountDerivatives();
	}

	public void CountDerivatives(EList<double> expectedOutput) {
		for (int i = 0; i < Output.Count; i++)
			Output[i].Derivative = expectedOutput[i] - Output[i].Value;
	}

	public EList<double> GetInputValues() {
		EList<double> values = new EList<double>();

		foreach (Unit unit in Input)
			values.Add(unit.Value);

		return values;
	}
	public EList<double> GetOutputValues() {
		EList<double> values = new EList<double>();

		foreach (Unit unit in Output)
			values.Add(unit.Value);

		return values;
	}
	
	public void FillWeightsRandom() { }
	public void FillBiasesRandom() { }
	public void ApplyDerivativesToWeights(double learningFactor) { }
	public void ApplyDerivativesToBiases(double learningFactor) { }
}

// public class StackPoolingLayer : ProcessingLayer, IInputStack, IOutputStack {
// 	public StackPoolingLayer(IOutputStack inputLayer, Filter mask, int stride, int method) {
// 		Model = new MatrixModel(inputLayer.Output.Count, stride);
// 		Mask = mask;
//
// 		List<StackUnit> nodes = new List<StackUnit>();
//
// 		for (int i = 0; i < Model.FilterOutputsCount(mask); i++) {
// 			List<StackUnit> units = new List<StackUnit>();
//
// 			for (int x = 0; x < Mask.Count(); x++) {
// 				int inner = x % Mask.Size + x / Mask.Size * Model.Size;
// 				int outer = i % Model.FilterLineCount(Mask) + i / Model.FilterLineCount(Mask) * Model.Size;
//
// 				units.Add(inputLayer.Output[inner + outer]);
// 			}
//
// 			nodes.Add(new StackPoolingNode(units, method, inputLayer.Output[0].Values.Count));
// 		}
//
// 		Input = nodes;
// 		Output = nodes;
// 	}
//
// 	public MatrixModel Model { get; set; }
// 	public Filter Mask { get; set; }
// 	public List<StackUnit> Input { get; set; }
//
// 	public override void Count() {
// 		foreach (StackPoolingNode unit in Output)
// 			unit.Count();
// 	}
//
// 	public override void CountDerivatives() {
// 		double part = 1d / Mask.Count();
//
// 		foreach (StackPoolingNode node in Input)
// 		foreach (StackUnit inputLayerUnit in node.InputUnits)
// 			for (int s = 0; s < inputLayerUnit.Derivatives.Count; s++)
// 				inputLayerUnit.Derivatives[s] += part * node.Derivatives[s];
// 	}
//
// 	public override void CountDerivatives(List<double> expectedOutput) {
// 		for (int i = 0; i < Output.Count; i++)
// 		for (int s = 0; s < Output[0].Values.Count; s++)
// 			Output[i].Derivatives[s] = expectedOutput[i] - Output[i].Values[s];
// 	}
//
// 	public List<StackUnit> Output { get; set; }
// }

}