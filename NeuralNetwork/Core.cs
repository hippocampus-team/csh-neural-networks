using System;
using System.Linq;

namespace NeuralNetworks
{
    public class Neuron
    {
        public double value { get; set; }
        public Neuron[] inputNeurons { get; set; }
        public double[] Weights { get; set; }
        public double bias { get; set; }
        public bool isActive { get; set; }

        public void Count()
        {
            if (isActive)
            {
                double WeightedSum = 0;

                for (int i = 0; i < inputNeurons.Length; i++)
                {
                    WeightedSum += inputNeurons[i].value * Weights[i];
                }

                value = Tools.Sigmoid(WeightedSum - bias);
            }
            else
                value = 0;
        }

        public Neuron(Neuron[] _inputNeurons)
        {
            value = 0;
            inputNeurons = _inputNeurons;
            Weights = new double[_inputNeurons.Length];
            bias = Constants.defaultBias;
            isActive = true;
        }

        public Neuron(Neuron[] _inputNeurons, double _bias)
        {
            value = 0;
            inputNeurons = _inputNeurons;
            Weights = new double[_inputNeurons.Length];
            bias = _bias;
            isActive = true;
        }

        public Neuron(Neuron[] _inputNeurons, double[] _Weights)
        {
            value = 0;
            inputNeurons = _inputNeurons;
            Weights = _Weights;
            bias = Constants.defaultBias;
            isActive = true;
        }

        public Neuron()
        {
            value = 0;
            inputNeurons = new Neuron[0];
            Weights = new double[0];
            bias = Constants.defaultBias;
            isActive = true;
        }

        public Neuron(double _bias)
        {
            value = 0;
            inputNeurons = new Neuron[0];
            Weights = new double[0];
            bias = _bias;
            isActive = true;
        }
    }
    
    public class Layer
    {
        public Neuron[] Neurons { get; set; }

        public void Count()
        {
            foreach (Neuron neuron in Neurons)
            {
                neuron.Count();
            }
        }

        public Layer(int n, Layer prevLayer)
        {
            Neuron[] _Neurons = new Neuron[n];

            for (int i = 0; i < n; i++)
            {
                _Neurons[i] = new Neuron(prevLayer.Neurons);
            }

            Neurons = _Neurons;
        }

        public Layer(int n)
        {
            Neuron[] _Neurons = new Neuron[n];

            for (int i = 0; i < n; i++)
            {
                _Neurons[i] = new Neuron();
            }

            Neurons = _Neurons;
        }

        public Neuron this[int neuron]
        {
            get
            {
                return Neurons[neuron];
            }

            set
            {
                Neurons[neuron] = value;
            }
        }
    }

    public class NeuralNetwork
    {
        public Layer[] Layers { get; }
        public double cost { get; set; }

        public int[] size { get; }
        public int neuronsCount { get; }

        public bool useBias { get; set; }

        public void Count()
        {
            for (int i = 1; i < Layers.Length; i++)
            {
                Layers[i].Count();
            }
        }

        public void RandomWeightsFill()
        {
            Random rnd = new Random(Guid.NewGuid().GetHashCode());

            foreach (Layer layer in Layers)
                foreach (Neuron neuron in layer.Neurons)
                    for (int i = 0; i < neuron.Weights.Length; i++)
                        neuron.Weights[i] = rnd.NextDouble() * 4 - 2;
        }

        public void RandomWeightsMutation(int rate)
        {
            Random rnd = new Random(Guid.NewGuid().GetHashCode());

            for (int i = 0; i < rate; i++)
            {
                Layer layer = Layers[rnd.Next(1, size.Length)];
                Neuron neuron = layer.Neurons[rnd.Next(layer.Neurons.Length)];
                neuron.Weights[rnd.Next(neuron.Weights.Length)] += rnd.NextDouble() * 4 - 2;
            }

            for (int i = 1; i < size.Length; i++)
                for (int j = 0; j < Layers[i].Neurons.Length; j++)
                    for (int k = 0; k < this[i, j].Weights.Length; k++)
                    {
                        if (this[i, j].Weights[k] > Constants.weightMutationMax)
                            this[i, j].Weights[k] = Constants.weightMutationMax;
                        if (this[i, j].Weights[k] < Constants.weightMutationMin)
                            this[i, j].Weights[k] = Constants.weightMutationMin;
                    }

        }

        public void RandomWeightsMutation(double percent)
        {
            Random rnd = new Random(Guid.NewGuid().GetHashCode());

            for (int i = 0; i < Math.Ceiling(neuronsCount / 100d * percent); i++)
            {
                Layer layer = Layers[rnd.Next(1, Layers.Length)];
                Neuron neuron = layer[rnd.Next(layer.Neurons.Length)];
                neuron.Weights[rnd.Next(neuron.Weights.Length)] += rnd.NextDouble() * 4 - 2;
            }

            for (int i = 1; i < size.Length; i++)
                for (int j = 0; j < Layers[i].Neurons.Length; j++)
                    for (int k = 0; k < this[i, j].Weights.Length; k++)
                    {
                        if (this[i, j].Weights[k] > Constants.weightMutationMax)
                            this[i, j].Weights[k] = Constants.weightMutationMax;
                        if (this[i, j].Weights[k] < Constants.weightMutationMin)
                            this[i, j].Weights[k] = Constants.weightMutationMin;
                    }

        }

        public void RandomBiasesFill()
        {
            Random rnd = new Random(Guid.NewGuid().GetHashCode());

            foreach (Layer layer in Layers)
                foreach (Neuron neuron in layer.Neurons)
                    neuron.bias = rnd.NextDouble() * Constants.maxRandomBias;
        }

        public void RandomBiasesFill(double max)
        {
            Random rnd = new Random(Guid.NewGuid().GetHashCode());

            foreach (Layer layer in Layers)
                foreach (Neuron neuron in layer.Neurons)
                    neuron.bias = rnd.NextDouble() * max;
        }

        public NeuralNetwork(int[] layersWeight)
        {
            useBias = true;

            Layer[] _Layers = new Layer[layersWeight.Length];
            _Layers[0] = new Layer(layersWeight[0]);

            for (int i = 1; i < layersWeight.Length; i++)
                _Layers[i] = new Layer(layersWeight[i], _Layers[i - 1]);

            Layers = _Layers;
            cost = 0;
            size = layersWeight;

            int _neuronsCount = 0;
            for (int i = 1; i < layersWeight.Length; i++)
                _neuronsCount += layersWeight[i] * layersWeight[i - 1];
            neuronsCount = _neuronsCount;
        }

        public static NeuralNetwork Duplicate(NeuralNetwork original)
        {
            NeuralNetwork _network = new NeuralNetwork(original.size);

            for (int i = 0; i < original.size.Length; i++)
                for (int j = 0; j < original.size[i]; j++)
                    for (int k = 0; k < original[i, j].Weights.Length; k++)
                        _network[i, j].Weights[k] = original[i, j].Weights[k];

            return _network;
        }

        public Neuron this [int layer, int neuron]
        {
            get
            {
                return Layers[layer].Neurons[neuron];
            }
        }

        public void Backpropagation(double[] expectedOutput, double learningFactor)
        {
            if (expectedOutput.Length != size.Last())
                throw new Exception("\"Expected output\" array is too small or too large.");

            double[,] derivatives = new double[size.Length, size.Max()];

            for (int i = 0; i < size.Last(); i++)
                derivatives[size.Length - 1, i] = (expectedOutput[i] - this[size.Length - 1, i].value) * learningFactor;

            for (int l = size.Length - 2; l > 0; l--)
                for (int i = 0; i < Layers[l].Neurons.Length; i++)
                {
                    double sum = 0;

                    for (int j = 0; j < Layers[l + 1].Neurons.Length; j++)
                        sum += derivatives[l + 1, j] * Tools.SigmoidDerivative(this[l + 1, j].value) * this[l + 1, j].Weights[i];

                    derivatives[l, i] = sum * learningFactor;
                }

            if (useBias)
                for (int i = 1; i < size.Length; i++)
                    for (int j = 0; j < Layers[i].Neurons.Length; j++)
                        this[i, j].bias += derivatives[i, j] * Tools.SigmoidDerivative(this[i, j].value);

            for (int i = 1; i < size.Length; i++)
                for (int j = 0; j < Layers[i].Neurons.Length; j++)
                    for (int k = 0; k < this[i, j].Weights.Length; k++)
                        this[i, j].Weights[k] += derivatives[i, j] * this[i - 1, k].value * Tools.SigmoidDerivative(this[i, j].value);
        }

        public void CarefulBackpropagation(double[] expectedOutput, double learningFactor)
        {
            if (expectedOutput.Length != size.Last())
                throw new Exception("\"Expected output\" array is too small or too large.");

            double[,] derivatives = new double[size.Length, size.Max()];

            for (int i = 0; i < size.Last(); i++)
                derivatives[size.Length - 1, i] = (expectedOutput[i] - this[size.Length - 1, i].value) * learningFactor;

            for (int l = size.Length - 2; l > 0; l--)
                for (int i = 0; i < Layers[l].Neurons.Length; i++)
                    if (this[l, i].isActive == true)
                    {
                        double sum = 0;

                        for (int j = 0; j < Layers[l + 1].Neurons.Length; j++)
                            sum += derivatives[l + 1, j] * Tools.SigmoidDerivative(this[l + 1, j].value) * this[l + 1, j].Weights[i];

                        derivatives[l, i] = sum * learningFactor;
                    }
                    else
                        derivatives[l, i] = 0;

            if (useBias)
                for (int i = 1; i < size.Length; i++)
                    for (int j = 0; j < Layers[i].Neurons.Length; j++)
                        this[i, j].bias += derivatives[i, j] * Tools.SigmoidDerivative(this[i, j].value);

            for (int i = 1; i < size.Length; i++)
                for (int j = 0; j < Layers[i].Neurons.Length; j++)
                    for (int k = 0; k < this[i, j].Weights.Length; k++)
                        this[i, j].Weights[k] += derivatives[i, j] * this[i - 1, k].value * Tools.SigmoidDerivative(this[i, j].value);
        }

        public void DisableRandomNeuron()
        {
            Random rnd = new Random(Guid.NewGuid().GetHashCode());

            int layer = rnd.Next(size.Length - 2) + 1;
            int neuron = rnd.Next(size[layer]);

            while (this[layer, neuron].isActive == false)
            {
                neuron = rnd.Next(size[layer]);
            }

            this[layer, neuron].isActive = false;
            this[layer, neuron].value = 0;
        }

        public void DisableRandomNeuronInEachLayer()
        {
            Random rnd = new Random(Guid.NewGuid().GetHashCode());

            for (int layer = 1; layer < size.Length - 1; layer++)
            {
                int neuron = rnd.Next(size[layer]);

                while (this[layer, neuron].isActive == false)
                {
                    neuron = rnd.Next(size[layer]);
                }

                this[layer, neuron].isActive = false;
                this[layer, neuron].value = 0;
            }
        }
    }

    public static class Tools
    {
        public static float Sigmoid(double value)
        {
            float k = (float)Math.Exp(value);
            return k / (1.0f + k);
        }

        public static double SigmoidDerivative(double value)
        {
            return value * (1 - value);
        }
    }

    public static class Constants
    {
        public const double defaultBias = 5;
        public const double maxRandomBias = 3;

        public const double weightMutationMax = 5;
        public const double weightMutationMin = 5;
    }
}