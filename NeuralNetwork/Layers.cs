using System;
using System.Collections.Generic;

namespace NeuralNetworks
{
    public interface ILayer
    {
        void Count();

        void FillWeightsRandom();
        void FillBiasesRandom();

        void CountDerivatives();
        void CountDerivatives(List<double> expectedOutput);

        void ApplyDerivativesToWeights(double learningFactor);
        void ApplyDerivativesToBiases(double learningFactor);
    }

    public interface IInputMono : ILayer
    {
        List<Unit> Input { get; set; }

        List<double> GetInputValues();
    }

    public interface IInputStack : ILayer
    {
        List<StackUnit> Input { get; set; }
    }

    public interface IOutputMono : ILayer
    {
        List<Unit> Output { get; set; }

        List<double> GetOutputValues();
    }

    public interface IOutputStack : ILayer
    {
        List<StackUnit> Output { get; set; }
    }

    public class DenceLayer : ILayer, IInputMono, IOutputMono
    {
        public List<Unit> Input { get; set; }
        public List<Unit> Output { get; set; }

        public void Count()
        {
            foreach (Unit unit in Output)
                unit.Count();
        }

        public void FillWeightsRandom()
        {
            Random rnd = new Random(Guid.NewGuid().GetHashCode());

            foreach (Neuron neuron in Output)
                for (int i = 0; i < neuron.Weights.Count; i++)
                    neuron.Weights[i] = (rnd.NextDouble() - 0.5) * Constants.weightRandomFillSpread;
        }

        public void FillBiasesRandom()
        {
            Random rnd = new Random(Guid.NewGuid().GetHashCode());

            foreach (Neuron neuron in Output)
                neuron.Bias = (rnd.NextDouble() - 0.5) * Constants.biasRandomFillSpread;
        }

        public void CountDerivatives()
        {
            foreach (Unit unit in Output)
                unit.CountDerivatives();
        }

        public void CountDerivatives(List<double> expectedOutput)
        {
            for (int i = 0; i < Output.Count; i++)
                Output[i].Derivative = expectedOutput[i] - Output[i].Value;
        }

        public void ApplyDerivativesToWeights(double learningFactor)
        {
            foreach (Unit unit in Output)
                unit.ApplyDerivativesToWeights(learningFactor);
        }

        public void ApplyDerivativesToBiases(double learningFactor)
        {
            foreach (Unit unit in Output)
                unit.ApplyDerivativesToBias(learningFactor);
        }

        //public void MutateRandomWeights()
        //{
        //    Random rnd = new Random(Guid.NewGuid().GetHashCode());

        //    int n = rnd.Next(Output.Count);
        //    Neuron neuron = (Neuron)Output[n];

        //    int w = rnd.Next(neuron.Weights.Count);
        //    neuron.Weights[w] += (rnd.NextDouble() - 0.5) * Constants.weightMutationSpread;

        //    neuron.Weights[w] = Tools.Clamp(neuron.Weights[w], Constants.weightMutatedMin, Constants.weightMutatedMax);
        //}

        //public void MutateRandomBiases()
        //{
        //    Random rnd = new Random(Guid.NewGuid().GetHashCode());

        //    int n = rnd.Next(Output.Count);
        //    Neuron neuron = (Neuron)Output[n];

        //    neuron.Bias += (rnd.NextDouble() - 0.5) * Constants.biasMutationSpread;

        //    neuron.Bias = Tools.Clamp(neuron.Bias, Constants.biasMutatedMin, Constants.biasMutatedMax);
        //}

        public List<double> GetInputValues()
        {
            List<double> values = new List<double>();

            foreach (Unit unit in Input)
                values.Add(unit.Value);

            return values;
        }

        public List<double> GetOutputValues()
        {
            List<double> values = new List<double>();

            foreach (Unit unit in Output)
                values.Add(unit.Value);

            return values;
        }

        public DenceLayer(int n, IOutputMono inputLayer)
        {
            List<Unit> neurons = new List<Unit>();

            for (int i = 0; i < n; i++)
                neurons.Add(new Neuron(inputLayer.Output));

            Input = neurons;
            Output = neurons;
        }
    }

    public class ConvolutionalLayer : ILayer, IInputStack, IOutputStack
    {
        public List<StackUnit> Input { get; set; }
        public List<StackUnit> Output { get; set; }

        public List<List<Filter>> Kernels { get; set; }
        public MatrixModel Model { get; set; }

        public void Count()
        {
            foreach (StackUnit unit in Output)
                unit.Count();
        }

        public void FillWeightsRandom()
        {
            Random rnd = new Random(Guid.NewGuid().GetHashCode());

            for (int i = 0; i < Kernels.Count; i++)
                for (int j = 0; j < Kernels[i].Count; j++)
                    for (int k = 0; k < Kernels[i][j].Values.Count; k++)
                        Kernels[i][j].Values[k] = (rnd.NextDouble() - 0.5) * Constants.weightRandomFillSpread;
        }

        public void FillBiasesRandom()
        {
            Random rnd = new Random(Guid.NewGuid().GetHashCode());

            foreach (StackUnit unit in Output)
                unit.Bias = (rnd.NextDouble() - 0.5) * Constants.biasRandomFillSpread;
        }

        public void CountDerivatives()
        {
            foreach (StackUnit unit in Output)
                unit.CountDerivatives();
        }

        public void CountDerivatives(List<double> expectedOutput)
        {
            for (int i = 0; i < Output.Count; i++)
                for (int s = 0; s < Output[0].Values.Count; s++)
                    Output[i].Derivatives[s] = expectedOutput[i] - Output[i].Values[s];
        }

        public void ApplyDerivativesToWeights(double learningFactor)
        {
            foreach (StackUnit unit in Output)
                unit.ApplyDerivativesToWeights(learningFactor);
        }

        public void ApplyDerivativesToBiases(double learningFactor)
        {
            foreach (StackUnit unit in Output)
                unit.ApplyDerivativesToBias(learningFactor);
        }

        public ConvolutionalLayer(IOutputStack inputLayer, Filter filter, int filtersAmount, int stride)
        {
            Kernels = new List<List<Filter>>();
            Model = new MatrixModel(inputLayer.Output, stride);

            for (int i = 0; i < filtersAmount; i++)
            {
                Kernels.Add(new List<Filter>());

                for (int j = 0; j < inputLayer.Output[i].Values.Count; j++)
                    Kernels[i].Add((Filter)filter.Clone());
            }

            List<StackUnit> neurons = new List<StackUnit>();

            for (int i = 0; i < Model.FilterOutputsCount(filter); i++)
            {
                List<int> indexes = new List<int>();

                for (int x = 0; x < filter.Count(); x++)
                {
                    int inner = x % filter.Size + (x / filter.Size) * Model.Size;
                    int outer = i % Model.FilterLineCount(filter) + (i / Model.FilterLineCount(filter)) * Model.Size;

                    indexes.Add(inner + outer);
                }

                neurons.Add(new ConvolutionalNeuron(inputLayer.Output, Kernels, indexes, filtersAmount));
            }

            Input = neurons;
            Output = neurons;
        }
    }

    public abstract class ProcessingLayer : ILayer
    {
        public abstract void Count();

        public void FillWeightsRandom() { }
        public void FillBiasesRandom() { }

        public abstract void CountDerivatives();
        public abstract void CountDerivatives(List<double> expectedOutput);

        public void ApplyDerivativesToWeights(double learningFactor) { }
        public void ApplyDerivativesToBiases(double learningFactor) { }
    }

    public class SimpleLayer : ProcessingLayer, IInputMono, IOutputMono
    {
        public List<Unit> Input { get; set; }
        public List<Unit> Output { get; set; }

        public override void Count()
        {
            foreach (Node unit in Output)
                unit.Count();
        }

        public override void CountDerivatives()
        {
            foreach (Unit unit in Input)
                unit.CountDerivatives();
        }

        public override void CountDerivatives(List<double> expectedOutput)
        {
            for (int i = 0; i < Output.Count; i++)
                Output[i].Derivative = expectedOutput[i] - Output[i].Value;
        }

        public SimpleLayer(int n)
        {
            List<Unit> nodes = new List<Unit>();

            for (int i = 0; i < n; i++)
                nodes.Add(new Node());

            Input = nodes;
            Output = nodes;
        }

        public SimpleLayer(double[] values)
        {
            List<Unit> nodes = new List<Unit>();

            for (int i = 0; i < values.Length; i++)
                nodes.Add(new Node(values[i]));

            Input = nodes;
            Output = nodes;
        }

        public SimpleLayer(IOutputMono inputLayer)
        {
            List<Unit> nodes = new List<Unit>();

            for (int i = 0; i < inputLayer.Output.Count; i++)
                nodes.Add(new ReferNode(inputLayer.Output[i]));

            Input = nodes;
            Output = nodes;
        }

        public List<double> GetInputValues()
        {
            List<double> values = new List<double>();

            foreach (Unit unit in Input)
                values.Add(unit.Value);

            return values;
        }

        public List<double> GetOutputValues()
        {
            List<double> values = new List<double>();

            foreach (Unit unit in Output)
                values.Add(unit.Value);

            return values;
        }
    }

    //TODO: StackSimpleLayer

    public class StackPoolingLayer : ProcessingLayer, IInputStack, IOutputStack
    {
        public List<StackUnit> Input { get; set; }
        public List<StackUnit> Output { get; set; }

        public MatrixModel Model { get; set; }
        public Filter Mask { get; set; }

        public override void Count()
        {
            foreach (StackPoolingNode unit in Output)
                unit.Count();
        }

        public override void CountDerivatives()
        {
            double part = 1d / Mask.Count();

            foreach (StackPoolingNode node in Input)
                foreach (StackUnit inputLayerUnit in node.InputUnits)
                    for (int s = 0; s < inputLayerUnit.Derivatives.Count; s++)
                        inputLayerUnit.Derivatives[s] += part * node.Derivatives[s];
        }

        public override void CountDerivatives(List<double> expectedOutput)
        {
            for (int i = 0; i < Output.Count; i++)
                for (int s = 0; s < Output[0].Values.Count; s++)
                    Output[i].Derivatives[s] = expectedOutput[i] - Output[i].Values[s];
        }

        public StackPoolingLayer(IOutputStack inputLayer, Filter mask, int stride, int method)
        {
            Model = new MatrixModel(inputLayer.Output.Count, stride);
            Mask = mask;

            List<StackUnit> nodes = new List<StackUnit>();

            for (int i = 0; i < Model.FilterOutputsCount(mask); i++)
            {
                List<StackUnit> units = new List<StackUnit>();

                for (int x = 0; x < Mask.Count(); x++)
                {
                    int inner = x % Mask.Size + (x / Mask.Size) * Model.Size;
                    int outer = i % Model.FilterLineCount(Mask) + (i / Model.FilterLineCount(Mask)) * Model.Size;

                    units.Add(inputLayer.Output[inner + outer]);
                }

                nodes.Add(new StackPoolingNode(units, method, inputLayer.Output[0].Values.Count));
            }

            Input = nodes;
            Output = nodes;
        }
    }

    public class FlattenLayer : ProcessingLayer, IInputStack, IOutputMono
    {
        public List<StackUnit> Input { get; set; }
        public List<Unit> Output { get; set; }

        public override void Count()
        {
            //Skip Input because it's StackRefer (No Count() requiered)

            foreach (Unit unit in Output)
                unit.Count();
        }

        public override void CountDerivatives()
        {
            int stack = Input[0].Derivatives.Count;

            for (int i = 0; i < Input.Count; i++)
                for (int s = 0; s < Input[i].Derivatives.Count; s++)
                    Input[i].Derivatives[s] = Output[i * stack + s].Derivative;
        }

        public override void CountDerivatives(List<double> expectedOutput)
        {
            for (int i = 0; i < Output.Count; i++)
                Output[i].Derivative = expectedOutput[i] - Output[i].Value;

            CountDerivatives();
        }

        public FlattenLayer(IOutputStack inputLayer)
        {
            Input = new List<StackUnit>();
            Output = new List<Unit>();

            foreach (StackUnit unit in inputLayer.Output)
                Input.Add(new StackReferNode(unit));

            int count = Input.Count * Input[0].Values.Count;

            for (int i = 0; i < Input.Count; i++)
                for (int j = 0; j < Input[i].Values.Count; j++)
                    Output.Add(new ArrayReferNode(Input[i], j));
        }

        public List<double> GetOutputValues()
        {
            List<double> values = new List<double>();

            foreach (Unit unit in Output)
                values.Add(unit.Value);

            return values;
        }
    }
}