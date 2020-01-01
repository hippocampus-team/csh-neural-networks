using System;
using System.Collections.Generic;

namespace NeuralNetworks
{
    public abstract class Layer
    {
        public List<Unit> Input { get; set; }
        public List<Unit> Output { get; set; }

        public void Count()
        {
            foreach (Unit unit in Output)
                unit.Count();
        }

        public double[] GetValues()
        {
            double[] values = new double[Output.Count];

            for (int i = 0; i < Output.Count; i++)
                values[i] = Output[i].Value;

            return values;
        }

        public abstract void FillWeightsRandom();
        public abstract void FillBiasesRandom();

        public abstract void MutateRandomWeights();
        public abstract void MutateRandomBiases();

        public abstract void CountDerivatives();
        public void CountDerivatives(double[] expectedOutput)
        {
            for (int i = 0; i < Output.Count; i++)
                Output[i].Derivative = expectedOutput[i] - Output[i].Value;
        }

        public abstract void ApplyDerivativesToWeights(double learningFactor);
        public abstract void ApplyDerivativesToBiases(double learningFactor);

        public Unit this[int index]
        {
            get
            {
                return Output[index];
            }

            set
            {
                Output[index] = value;
            }
        }
    }

    public class SimpleLayer : Layer
    {
        public override void FillWeightsRandom() { }

        public override void FillBiasesRandom() { }

        public override void MutateRandomWeights() { }

        public override void MutateRandomBiases() { }

        public SimpleLayer(int n)
        {
            List<Unit> _nodes = new List<Unit>();

            for (int i = 0; i < n; i++)
                _nodes.Add(new Node());

            Input = _nodes;
            Output = _nodes;
        }

        public SimpleLayer(double[] values)
        {
            List<Unit> _nodes = new List<Unit>();

            for (int i = 0; i < values.Length; i++)
                _nodes.Add(new Node(values[i]));

            Input = _nodes;
            Output = _nodes;
        }

        public SimpleLayer(List<Unit> refers)
        {
            List<Unit> _nodes = new List<Unit>();

            for (int i = 0; i < refers.Count; i++)
                _nodes.Add(new ReferNode(refers[i]));

            Input = _nodes;
            Output = _nodes;
        }

        public override void CountDerivatives()
        {
            for (int i = 0; i < Output.Count; i++)
                Output[i].Derivative = Output[i].OutputUnits[i].Derivative;
        }

        public override void ApplyDerivativesToWeights(double learningFactor) { }

        public override void ApplyDerivativesToBiases(double learningFactor) { }
    }

    public class DenceLayer : Layer
    {
        public new void Count()
        {
            foreach (Neuron neuron in Output)
                neuron.Count();
        }

        public override void FillWeightsRandom()
        {
            Random rnd = new Random(Guid.NewGuid().GetHashCode());

            foreach (Neuron neuron in Output)
                for (int i = 0; i < neuron.Weights.Count; i++)
                    neuron.Weights[i] = (rnd.NextDouble() - 0.5) * Constants.weightRandomFillSpread;
        }

        public override void FillBiasesRandom()
        {
            Random rnd = new Random(Guid.NewGuid().GetHashCode());

            foreach (Neuron neuron in Output)
                neuron.Bias = (rnd.NextDouble() - 0.5) * Constants.biasRandomFillSpread;
        }

        public override void MutateRandomWeights()
        {
            Random rnd = new Random(Guid.NewGuid().GetHashCode());

            int n = rnd.Next(Output.Count);
            Neuron neuron = (Neuron)Output[n];

            int w = rnd.Next(neuron.Weights.Count);
            neuron.Weights[w] += (rnd.NextDouble() - 0.5) * Constants.weightMutationSpread;

            Output[n].Weights[w] = Tools.Clamp(Output[n].Weights[w], Constants.weightMutatedMin, Constants.weightMutatedMax);
        }

        public override void MutateRandomBiases()
        {
            Random rnd = new Random(Guid.NewGuid().GetHashCode());

            int n = rnd.Next(Output.Count);
            Neuron neuron = (Neuron)Output[n];

            neuron.Bias += (rnd.NextDouble() - 0.5) * Constants.biasMutationSpread;

            neuron.Bias = Tools.Clamp(neuron.Bias, Constants.biasMutatedMin, Constants.biasMutatedMax);
        }

        public override void CountDerivatives()
        {
            for (int i = 0; i < Output.Count; i++)
            {
                double der = 0;

                for (int j = 0; j < Output[i].OutputUnits.Count; j++)
                    der += Output[i].OutputUnits[j].Derivative * Tools.SigmoidDerivative(Output[i].OutputUnits[j].Value) * Output[i].OutputUnits[j].Weights[i];

                Output[i].Derivative = der;
            }
        }

        public override void ApplyDerivativesToWeights(double learningFactor)
        {
            for (int i = 0; i < Output.Count; i++)
                for (int j = 0; j < Output[i].Weights.Count; j++)
                    Output[i].Weights[j] += Output[i].Derivative * Output[i].InputUnits[j].Value * Tools.SigmoidDerivative(Output[i].Value) * learningFactor;
        }

        public override void ApplyDerivativesToBiases(double learningFactor)
        {
            for (int i = 0; i < Output.Count; i++)
                ((Neuron)Output[i]).Bias += Output[i].Derivative * Tools.SigmoidDerivative(Output[i].Value) * learningFactor;
        }

        public DenceLayer(int n, Layer prevLayer)
        {
            List<Unit> _neurons = new List<Unit>();

            for (int i = 0; i < n; i++)
                _neurons.Add(new Neuron(prevLayer.Output));

            Input = _neurons;
            Output = _neurons;
        }

        public DenceLayer(int n)
        {
            List<Unit> _neurons = new List<Unit>();

            for (int i = 0; i < n; i++)
                _neurons.Add(new Neuron());

            Input = _neurons;
            Output = _neurons;
        }
    }
}