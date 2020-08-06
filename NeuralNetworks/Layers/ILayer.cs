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