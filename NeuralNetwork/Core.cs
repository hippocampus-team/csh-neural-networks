using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks
{
    public class NeuralNetwork : ICloneable
    {
        public List<Layer> Layers { get; }

        private int Length;

        public bool UseBiases { get; set; }

        public void Count()
        {
            foreach (Layer layer in Layers)
                layer.Count();
        }

        public double GetCost(double expectedOutput, int index)
        {
            double cost = 1 - Math.Pow(expectedOutput - this[Length - 1][index].Value, 2);

            return cost;
        }

        public double[] GetCosts(double[] expectedOutput)
        {
            double[] costs = new double[this[Length - 1].Output.Count];
            for (int i = 0; i < costs.Length; i++)
                costs[i] = 1 - Math.Pow(expectedOutput[i] - this[Length - 1][i].Value, 2);

            return costs;
        }

        public double GetTotalCost(double[] expectedOutput)
        {
            double[] costs = GetCosts(expectedOutput);
            return costs.Average();
        }

        public void FillRandomWeights()
        {
            foreach (Layer layer in Layers)
                layer.FillWeightsRandom();
        }

        public void FillRandomBiases()
        {
            foreach (Layer layer in Layers)
                layer.FillBiasesRandom();
        }

        public void MutateRandomWeights(int rate)
        {
            Random rnd = new Random(Guid.NewGuid().GetHashCode());

            for (int i = 0; i < rate; i++)
                Layers[rnd.Next(1, Length)].MutateRandomWeights();
        }

        public void MutateRandomBiases(int rate)
        {
            Random rnd = new Random(Guid.NewGuid().GetHashCode());

            for (int i = 0; i < rate; i++)
                Layers[rnd.Next(1, Length)].MutateRandomBiases();
        }

        public void AddSimpleLayer(int length)
        {
            SimpleLayer layer;

            if (Length == 0)
                layer = new SimpleLayer(length);
            else
                layer = new SimpleLayer(this[Length - 1].Output);

            Length++;
            Layers.Add(layer);
        }

        public void AddDenceLayer(int length)
        {
            DenceLayer layer;

            if (Length == 0)
                layer = new DenceLayer(length);
            else
                layer = new DenceLayer(length, this[Length - 1]);

            Length++;
            Layers.Add(layer);
        }

        public NeuralNetwork()
        {
            Layers = new List<Layer>();
            UseBiases = true;
        }

        public Layer this [int layer]
        {
            get
            {
                return Layers[layer];
            }
        }

        public void Backpropagation(double[] expectedOutput, double learningFactor)
        {
            Layers[Length - 1].CountDerivatives(expectedOutput);

            for (int i = Length - 2; i > 0; i--)
                Layers[i].CountDerivatives();

            if (UseBiases)
                for (int i = Length - 1; i > 0; i--)
                    Layers[i].ApplyDerivativesToBiases(learningFactor);

            for (int i = Length - 1; i > 0; i--)
                Layers[i].ApplyDerivativesToWeights(learningFactor);
        }

        public object Clone()
        {
            return new NeuralNetwork();
        }
    }
}