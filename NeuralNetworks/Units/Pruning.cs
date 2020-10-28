using NeuralNetworks.Misc;

namespace NeuralNetworks.Units {

public class PrunableNeuron : Neuron {
	private EList<double> weightsMemoryDerivitives;

	public PrunableNeuron(EList<Unit> inputUnits, ActivationFunction activationFunction) 
		: base(inputUnits, activationFunction) {
		weightsMemoryDerivitives = new EList<double>(inputUnits.columns);

		for (int i = 0; i < inputUnits.Count; i++) 
			weightsMemoryDerivitives.Add(0);
	}
	
	public override void countDerivatives() {
		for (int i = 0; i < inputUnits.Count; i++) {
			double derivedWeight = derivative * weights[i];
			inputUnits[i].derivative += derivedWeight * activationFunction.countDerivative(value);
			weightsMemoryDerivitives[i] += derivedWeight;
		}
	}

	public void prune(double vanishThreshold) {
		for (int i = 0; i < weights.Count; i++)
			if (weightsMemoryDerivitives[i] < vanishThreshold)
				removeConnection(i);
		
		clearMemory();
	}

	private void removeConnection(int i) {
		inputUnits.RemoveAt(i);
		weights.RemoveAt(i);
	}
	
	public void clearMemory() {
		weightsMemoryDerivitives.Clear();
		for (int i = 0; i < inputUnits.Count; i++) 
			weightsMemoryDerivitives.Add(0);
	}
}

}