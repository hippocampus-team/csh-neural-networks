using System;
using System.Collections.Generic;
using NeuralNetworks.Misc;
using NeuralNetworks.Units;
using Newtonsoft.Json.Linq;

namespace NeuralNetworks.Layers {

public class PoolingLayer : SameInputOutputLayer {
	public override IEnumerable<Unit> units => nodes;

	private readonly DepthList<PoolingNode> nodes;
	private MatrixModel model { get; }
	private Filter mask { get; }

	private PoolingLayer() { }

	public PoolingLayer(LayerConnection inputConnection, Filter mask, int stride, PoolingMethod method) {
		model = new MatrixModel(inputConnection.length, stride);
		this.mask = mask;

		nodes = new DepthList<PoolingNode>(inputConnection.depth);

		int filterOutputsCount = model.filterOutputsCount(mask);
		int filterLineCount = model.filterLineCount(mask);

		for (int d = 0; d < inputConnection.depth; d++)
		for (int i = 0; i < filterOutputsCount; i++) {
			List<Unit> inputUnits = new List<Unit>();

			for (int x = 0; x < mask.count; x++) {
				int inner = x % mask.size + x / mask.size * model.size;
				int outer = i % filterLineCount + i / filterLineCount * model.size;

				inputUnits.Add(inputConnection[inner + outer, d]);
			}

			nodes.Add(new PoolingNode(inputUnits, method));
		}

		input = new PoolingLayerConnection(nodes);
	}

	public override void count() {
		foreach (PoolingNode node in nodes)
			node.count();
	}

	public override void countDerivativesOfPreviousLayer() {
		foreach (PoolingNode node in nodes)
			node.countDerivativesOfInputUnits();
	}

	public override void fillParametersRandomly() { }
	public override void countDerivatives(List<double> expectedOutput) { }
	public override void applyDerivativesToParameters(double learningFactor) { }
	
	public override List<double> getInputValues() => throw new NotImplementedException();
	public override Layer fillFromJObject(JObject json) => throw new NotImplementedException();

	public static PoolingLayer getEmpty() => new PoolingLayer();
	
	private class PoolingLayerConnection : LayerConnection {
		private readonly DepthList<PoolingNode> nodes;
		
		public IEnumerable<Unit> enumerable => nodes.toList();

		public Unit this[int index] {
			get => nodes[index];
			set => nodes[index] = (PoolingNode) value;
		}
		
		public Unit this[int index, int depthIndex] {
			get => nodes[index, depthIndex];
			set => nodes[index, depthIndex] = (PoolingNode) value;
		}

		public int length => nodes.length;
		public int depth => nodes.depth;

		public PoolingLayerConnection(DepthList<PoolingNode> nodes) {
			this.nodes = nodes;
		}
	}
}

}