using NeuralNetworks.Units;

namespace NeuralNetworks {

public class ConstructionNeuronIndexer {
	public static ConstructionNeuronIndexer activeIndexer { get; private set; }
	
	private readonly NeuralNetwork neuralNetwork;

	private ConstructionNeuronIndexer(NeuralNetwork neuralNetwork) => 
		this.neuralNetwork = neuralNetwork;

	public Unit getUnitById(string id) {
		string[] idData = id.Split('_');
		int layerIndex = int.Parse(idData[0]);
		int neuronIndex = int.Parse(idData[1]);

		return neuralNetwork[layerIndex][neuronIndex];
	}

	public static void startConstruction(NeuralNetwork neuralNetwork) {
		activeIndexer = new ConstructionNeuronIndexer(neuralNetwork);
	}

	public static void endConstruction() {
		activeIndexer = null;
	}
}

}