using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks
{
    public class NeuralNetwork : ICloneable
    {
        public List<ILayer> Layers { get; }
        private int Length { get; set; }
        public bool UseBiases { get; set; }

        public void Count()
        {
            foreach (ILayer layer in Layers)
                layer.Count();
        }

        public double GetCost(double expectedOutput, int index)
        {
            double cost = 1 - Math.Pow(expectedOutput - ((IOutputMono)Layers.Last()).Output[index].Value, 2);

            return cost;
        }

        public List<double> GetCosts(List<double> expectedOutput)
        {
            List<double> costs = new List<double>();
            for (int i = 0; i < expectedOutput.Count; i++)
                costs.Add(1 - Math.Pow(expectedOutput[i] - ((IOutputMono)Layers.Last()).Output[i].Value, 2));

            return costs;
        }

        public double GetTotalCost(List<double> expectedOutput)
        {
            List<double> costs = GetCosts(expectedOutput);
            return costs.Average();
        }

        public void FillRandomWeights()
        {
            foreach (ILayer layer in Layers)
                layer.FillWeightsRandom();
        }

        public void FillRandomBiases()
        {
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

        public void AddSimpleLayer(int length)
        {
            SimpleLayer layer;

            if (Length == 0)
                layer = new SimpleLayer(length);
            else
                layer = new SimpleLayer((IOutputMono)Layers.Last());

            Length++;
            Layers.Add(layer);
        }

        public void AddDenceLayer(int length)
        {
            DenceLayer layer;

            if (Length == 0)
                throw new Exception("Dence layer can not be first one");
            else
                layer = new DenceLayer(length, (IOutputMono)Layers[Length - 1]);

            Length++;
            Layers.Add(layer);
        }

        public void AddConvolutionalLayer(Filter filter, int filtersAmount, int stride)
        {
            ConvolutionalLayer layer;

            if (Length == 0)
                throw new Exception("Convolitional layer can not be first one");
            else
                layer = new ConvolutionalLayer((IOutputStack)Layers.Last(), filter, filtersAmount, stride);

            Length++;
            Layers.Add(layer);
        }

        public void AddPoolingLayer(Filter filter, int stride, int method)
        {
            StackPoolingLayer layer;

            if (Length == 0)
                throw new Exception("Pooling layer can not be first one");
            else
                layer = new StackPoolingLayer((IOutputStack)Layers.Last(), filter, stride, method);

            Length++;
            Layers.Add(layer);
        }

        public NeuralNetwork() : this(true) { }
        public NeuralNetwork(bool useBiases)
        {
            Layers = new List<ILayer>();
            UseBiases = useBiases;
        }

        public ILayer this [int layer]
        {
            get
            {
                return Layers[layer];
            }
        }

        public void Backpropagation(List<double> expectedOutput, double learningFactor)
        {
            Layers[Length - 1].CountDerivatives(expectedOutput);

            for (int i = Length - 1; i > 1; i--)
                Layers[i].CountDerivatives();

            if (UseBiases)
                for (int i = Length - 1; i > 1; i--)
                    Layers[i].ApplyDerivativesToBiases(learningFactor);

            for (int i = Length - 1; i > 1; i--)
                Layers[i].ApplyDerivativesToWeights(learningFactor);
        }

        //TODO: Clone()
        public object Clone()
        {
            return new NeuralNetwork();
        }
    }
}