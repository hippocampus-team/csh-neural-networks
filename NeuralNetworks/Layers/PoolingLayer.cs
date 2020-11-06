using System;
using System.Collections.Generic;
using NeuralNetworks.Misc;
using NeuralNetworks.Units;

namespace NeuralNetworks.Layers {

public class PoolingLayer : Layer {
	public sealed override EList<Unit> input { get; protected set; }
	public sealed override EList<Unit> output { get; protected set; }
	public override IEnumerable<Unit> neurons => input;
	
	private MatrixModel model { get; }
	private Filter mask { get; }

	private PoolingLayer() { }

	public PoolingLayer(Layer inputLayer, Filter mask, int stride, PoolingNode.PoolingMethod method) {
		model = new MatrixModel(inputLayer.output, stride);
		this.mask = mask;

		EList<Unit> nodes = new EList<Unit>(inputLayer.output.columns);

		for (int c = 0; c < inputLayer.output.columns; c++)
		for (int i = 0; i < model.filterOutputsCount(mask); i++) {
			EList<Unit> inputUnits = new EList<Unit>();

			for (int x = 0; x < mask.count; x++) {
				int inner = x % mask.size + x / mask.size * model.size;
				int outer = i % model.filterLineCount(mask) + i / model.filterLineCount(mask) * model.size;

				inputUnits.Add(inputLayer.output[inner + outer, c]);
			}

			nodes.Add(new PoolingNode(inputUnits, method));
		}

		input = nodes;
		output = nodes;
	}

	public override void count() {
		foreach (Unit unit in output)
			unit.count();
	}

	public override void countDerivatives() {
		double part = 1d / mask.count;

		foreach (Unit node in input)
		foreach (Unit inputLayerUnit in node.inputUnits)
			inputLayerUnit.derivative += part * node.derivative;
	}

	public override void countDerivatives(EList<double> expectedOutput) {
		for (int r = 0; r < output.rows; r++)
		for (int c = 0; c < output.columns; c++)
			output[r, c].derivative = expectedOutput[r, c] - output[r, c].value;
	}

	public override EList<double> getInputValues() => throw new NotImplementedException();

	public override EList<double> getOutputValues() => throw new NotImplementedException();

	public override void fillWeightsRandom() { }
	public override void fillBiasesRandom() { }
	public override void applyDerivativesToWeights(double learningFactor) { }
	public override void applyDerivativesToBiases(double learningFactor) { }
	public static PoolingLayer getEmpty() => new PoolingLayer();
}

}