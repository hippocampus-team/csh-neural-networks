using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Misc;

namespace NeuralNetworks {
	public class NeuralNetwork {
		public NeuralNetwork() : this(true) { }

		public NeuralNetwork(bool useBiases) {
			Layers = new List<ILayer>();
			UseBiases = useBiases;
		}

		private List<ILayer> Layers { get; }
		private int Length { get; set; }
		private bool UseBiases { get; }

		public ILayer this[int layer] => Layers[layer];

		public void Run() {
			foreach (ILayer layer in Layers)
				layer.Count();
		}

		public double GetCost(double expectedOutput, int index) =>
			1 - Math.Pow(expectedOutput - (Layers.Last()).Output[index].Value, 2);

		public List<double> GetCosts(List<double> expectedOutput) =>
			expectedOutput.Select((t, i) =>
				1 - Math.Pow(t - Layers.Last().Output[i].Value, 2)).ToList();

		public double GetTotalCost(List<double> expectedOutput) =>
			GetCosts(expectedOutput).Average();

		public void FillRandomWeights() {
			foreach (ILayer layer in Layers)
				layer.FillWeightsRandom();
		}

		public void FillRandomBiases() {
			foreach (ILayer layer in Layers)
				layer.FillBiasesRandom();
		}

		//public void MutateRandomWeights(int rate)
		//{
		//    Random rnd = new Random(Guid.NewGuid().GetHashCode());

		//    for (int i = 0; i < rate; i++)
		//        Layers[rnd.Next(1, Length)].MutateRandomWeights();
		//}

		//public void MutateRandomBiases(int rate)
		//{
		//    Random rnd = new Random(Guid.NewGuid().GetHashCode());

		//    for (int i = 0; i < rate; i++)
		//        Layers[rnd.Next(1, Length)].MutateRandomBiases();
		//}

		public void PutData(IEnumerable<double> data) {
			IEnumerator<double> enumerator = data.GetEnumerator();
			
			foreach (Unit unit in Layers.First().Input) {
				unit.Value = enumerator.Current;
				enumerator.MoveNext();
			}
			
			enumerator.Dispose();
		}

		public void SetInputLength(int length) {
			if (Length > 0) Layers[0] = new SimpleLayer(length);
			else AddSimpleLayer(length);
		}

		public void AddSimpleLayer(int length) {
			SimpleLayer layer = Length == 0 ? new SimpleLayer(length) : new SimpleLayer(Layers.Last());

			Length++;
			Layers.Add(layer);
		}

		public void AddDenceLayer(int length) {
			if (Length == 0)
				throw new Exception("Dence layer can not be first one");

			Layers.Add(new DenceLayer(length, Layers.Last()));
			Length++;
		}

		public void AddConvolutionalLayer(Filter filter, int filtersAmount, int stride) {
			if (Length == 0)
				throw new Exception("Convolitional layer can not be first one");

			Layers.Add(new ConvolutionalLayer(Layers.Last(), filter, filtersAmount, stride));
			Length++;
		}

		// public void AddPoolingLayer(Filter filter, int stride, int method) {
		// 	if (Length == 0)
		// 		throw new Exception("Pooling layer can not be first one");
		// 	
		// 	Layers.Add(new StackPoolingLayer((IOutputStack) Layers.Last(), filter, stride, method));
		// 	Length++;
		// }

		public void Backpropagate(EList<double> expectedOutput, double learningFactor) {
			Layers.Last().CountDerivatives(expectedOutput);

			for (int i = Length - 1; i > 1; i--)
				Layers[i].CountDerivatives();

			for (int i = Length - 1; i > 1; i--)
				Layers[i].ApplyDerivativesToWeights(learningFactor);

			if (!UseBiases) return;
			for (int i = Length - 1; i > 1; i--)
				Layers[i].ApplyDerivativesToBiases(learningFactor);
		}
	}
}