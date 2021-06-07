using System;
using System.Collections.Generic;
using NeuralNetworks.Misc;
using NeuralNetworks.Units;
using Newtonsoft.Json.Linq;

namespace NeuralNetworks.Layers {

public class SimpleLayer : SameInputOutputLayer {
	public override IEnumerable<Unit> units => nodes;
	public override LayerType layerType => LayerType.simple;

	private readonly List<Node> nodes;

	public SimpleLayer(int n) {
		nodes = new List<Node>();

		for (int i = 0; i < n; i++)
			nodes.Add(new Node());

		input = new SimpleLayerConnection(nodes);
	}

	public SimpleLayer(IEnumerable<double> values) {
		nodes = new List<Node>();

		foreach (double value in values)
			nodes.Add(new Node(value));

		input = new SimpleLayerConnection(nodes);
	}

	public SimpleLayer(LayerConnection inputConnection) {
		nodes = new List<Node>();

		foreach (Unit unit in inputConnection.enumerable)
			nodes.Add(new ReferNode(unit));

		input = new SimpleLayerConnection(nodes);
	}

	public override void count() {
		foreach (Node node in nodes)
			node.count();
	}

	public override void countDerivativesOfPreviousLayer() {
		foreach (Node node in nodes)
			node.countDerivativesOfInputUnits();
	}

	public override void countDerivatives(List<double> expectedOutput) {
		for (int i = 0; i < nodes.Count; i++)
			nodes[i].countDerivative(expectedOutput[i]);
	}

	public override List<double> getInputValues() {
		List<double> values = new List<double>();

		foreach (Node node in nodes) 
			values.Add(node.value);

		return values;
	}

	public override void fillParametersRandomly() { }
	public override void applyDerivativesToParameters(double learningFactor) { }
	
	public override Layer fillFromJObject(JObject json) {
		JArray unitsJArray = json["units"]!.Value<JArray>();
		
		foreach (JToken unitToken in unitsJArray) {
			Unit unit = JFactory.constructUnit((JObject) unitToken);
			
			if (!(unit is Node)) throw new 
				ArgumentException("Only nodes are allowed in simple layers. Failed at " + unitToken.Path);
			
			nodes.Add((Node) unit);
		}
		
		return this;
	}
	
	public static SimpleLayer getEmpty() => new SimpleLayer(0);
	
	private class SimpleLayerConnection : NoDepthLayerConnection {
		private readonly List<Node> nodes;
		
		public override IEnumerable<Unit> enumerable => nodes;

		public override Unit this[int index] {
			get => nodes[index];
			set => nodes[index] = (Node) value;
		}

		public override int length => nodes.Count;

		public SimpleLayerConnection(List<Node> nodes) {
			this.nodes = nodes;
		}
	}
}

}