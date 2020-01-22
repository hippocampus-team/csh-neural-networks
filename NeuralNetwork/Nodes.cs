using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks
{
    public abstract class Unit : IUnit
    {
        public double Value { get; set; }
        public List<double> Weights { get; set; }
        public double Bias { get; set; }
        public double Derivative { get; set; }

        public List<Unit> InputUnits { get; set; }
        public List<Unit> OutputUnits { get; set; }

        public abstract void Count();
        public abstract void CountDerivatives();
        public abstract void ApplyDerivativesToWeights(double learningFactor);
        public abstract void ApplyDerivativesToBias(double learningFactor);
    }

    public class Node : Unit
    {
        public override void Count() { }
        public override void CountDerivatives() { }
        public override void ApplyDerivativesToWeights(double learningFactor) { }
        public override void ApplyDerivativesToBias(double learningFactor) { }

        public Node() : this(0) { }
        public Node(double value)
        {
            Value = value;
            Weights = new List<double>();
            Bias = 0;
            Derivative = 0;

            InputUnits = new List<Unit>();
            OutputUnits = new List<Unit>();
        }
    }

    public class ReferNode : Node
    {
        public override void Count()
        {
            Value = InputUnits[0].Value;
        }

        public override void CountDerivatives()
        {
            // Derivative = OutputUnits[0].Derivative;

            InputUnits[0].Derivative = Derivative;
        }

        public ReferNode(Unit inputUnit)
        {
            Value = inputUnit.Value;
            Weights = new List<double>();
            Bias = 0;
            Derivative = 0;

            Weights.Add(1);

            InputUnits = new List<Unit>();
            OutputUnits = new List<Unit>();

            InputUnits.Add(inputUnit);
            inputUnit.OutputUnits.Add(this);
        }
    }

    public class ArrayReferNode : Node
    {
        private List<double> ArrayRefer { get; set; }
        private int ArrayIndex { get; set; }

        public override void Count()
        {
            Value = ArrayRefer[ArrayIndex];
        }

        public override void CountDerivatives()
        {
            Derivative = OutputUnits[0].Derivative;
        }

        public ArrayReferNode(StackUnit stackRefer, int stackIndex) : this(stackRefer.Values, stackIndex) { }
        public ArrayReferNode(List<double> array, int index) : base(array[index])
        {
            Weights.Add(1);

            ArrayRefer = array;
            ArrayIndex = index;
        }
    }

    public class PoolingNode : Node
    {
        public int Method { get; set; }

        public override void Count()
        {
            switch (Method)
            {
                case METHOD_AVERAGE:
                    double average = 0;

                    foreach (Unit unit in InputUnits)
                        average += unit.Value;

                    Value = average / InputUnits.Count;
                    break;

                case METHOD_MAX:
                    double max = InputUnits[0].Value;

                    for (int i = 1; i < InputUnits.Count; i++)
                        if (InputUnits[i].Value > max)
                            max = InputUnits[i].Value;

                    Value = max;
                    break;

                case METHOD_MIN:
                    double min = InputUnits[0].Value;

                    for (int i = 1; i < InputUnits.Count; i++)
                        if (InputUnits[i].Value < min)
                            min = InputUnits[i].Value;

                    Value = min;
                    break;
            }
        }

        public const int METHOD_AVERAGE = 0;
        public const int METHOD_MAX = 1;
        public const int METHOD_MIN = 2;

        public PoolingNode(List<Unit> inputUnits) : this(inputUnits, 0) { }
        public PoolingNode(List<Unit> inputUnits, int method)
        {
            Value = 0;
            Weights = new List<double>();
            Bias = 0;
            Derivative = 0;

            Method = method;

            InputUnits = inputUnits;
            OutputUnits = new List<Unit>();

            foreach (Unit unit in inputUnits)
                unit.OutputUnits.Add(this);
        }
    }

    public class Neuron : Unit
    {
        public override void Count()
        {
            double weightedSum = 0;

            for (int i = 0; i < InputUnits.Count; i++)
                weightedSum += InputUnits[i].Value * Weights[i];

            Value = Tools.Sigmoid(weightedSum + Bias);
        }

        public override void CountDerivatives()
        {
            //double derivative = 0;

            //for (int i = 0; i < OutputUnits.Count; i++)
            //    derivative += OutputUnits[i].Derivative * Tools.SigmoidDerivative(OutputUnits[i].Value) * OutputUnits[i].Weights[i];

            //Derivative = derivative;

            for (int i = 0; i < InputUnits.Count; i++)
                InputUnits[i].Derivative += Derivative * Tools.SigmoidDerivative(Value) * Weights[i];
        }

        public override void ApplyDerivativesToWeights(double learningFactor)
        {
            for (int i = 0; i < Weights.Count; i++)
                Weights[i] += Derivative * InputUnits[i].Value * Tools.SigmoidDerivative(Value) * learningFactor;
        }

        public override void ApplyDerivativesToBias(double learningFactor)
        {
            Bias += Derivative * Tools.SigmoidDerivative(Value) * learningFactor;
        }

        public Neuron(List<Unit> inputUnits)
        {
            Value = 0;
            Weights = new List<double>();
            InputUnits = inputUnits;
            OutputUnits = new List<Unit>();
            Bias = Constants.defaultBias;

            for (int i = 0; i < inputUnits.Count; i++)
                Weights.Add(1);

            foreach (Unit unit in inputUnits)
                unit.OutputUnits.Add(this);
        }
    }

    public class SharedNeuron : Neuron
    {
        public List<double> SharedWeights { get; set; }
        public List<int> Indexes { get; set; }

        public override void Count()
        {
            double WeightedSum = 0;

            for (int i = 0; i < InputUnits.Count; i++)
                WeightedSum += InputUnits[i].Value * SharedWeights[Indexes[i]];

            Value = Tools.Sigmoid(WeightedSum - Bias);
        }

        public override void CountDerivatives()
        {
            //FetchWeights();
            //base.CountDerivatives();

            for (int i = 0; i < InputUnits.Count; i++)
                InputUnits[i].Derivative += Derivative * Tools.SigmoidDerivative(Value) * SharedWeights[Indexes[i]];
        }

        public override void ApplyDerivativesToWeights(double learningFactor)
        {
            for (int i = 0; i < Weights.Count; i++)
                SharedWeights[Indexes[i]] += Derivative * InputUnits[i].Value * Tools.SigmoidDerivative(Value) * learningFactor;
        }

        private void FetchWeights()
        {
            for (int i = 0; i < Weights.Count; i++)
                Weights[i] = SharedWeights[Indexes[i]];
        }

        public SharedNeuron(List<double> sharedWeights, List<int> indexes) : this(new List<Unit>(), sharedWeights, indexes) {}
        public SharedNeuron(List<Unit> inputUnits, List<double> sharedWeights, List<int> indexes) : base(inputUnits)
        {
            SharedWeights = sharedWeights;
            Indexes = indexes;

            FetchWeights();
        }
    }

    public abstract class StackUnit : IUnit
    {
        public List<double> Values { get; set; }
        public List<List<double>> Weights { get; set; }
        public List<double> Derivatives { get; set; }
        public double Bias { get; set; }

        public List<StackUnit> InputUnits { get; set; }
        public List<StackUnit> OutputUnits { get; set; }

        public abstract void Count();
        public abstract void CountDerivatives();
        public abstract void ApplyDerivativesToWeights(double learningFactor);
        public abstract void ApplyDerivativesToBias(double learningFactor);
    }

    public class StackNode : StackUnit
    {
        public override void Count() { }
        public override void CountDerivatives() { }
        public override void ApplyDerivativesToBias(double learningFactor) { }
        public override void ApplyDerivativesToWeights(double learningFactor) { }

        public StackNode() : this(0) { }
        public StackNode(int stackLength)
        {
            Values = new List<double>();
            Weights = new List<List<double>>();
            Derivatives = new List<double>();
            Bias = Constants.defaultBias;

            InputUnits = null;
            OutputUnits = new List<StackUnit>();

            for (int i = 0; i < stackLength; i++)
            {
                Values.Add(0);
                Derivatives.Add(0);
            }
        }
    }

    public class StackReferNode : StackNode
    {
        public StackReferNode(StackUnit inputUnit)
        {
            Values = inputUnit.Values;
            Derivatives = inputUnit.Derivatives;

            InputUnits = new List<StackUnit>();
            OutputUnits = new List<StackUnit>();

            InputUnits.Add(inputUnit);
            inputUnit.OutputUnits.Add(this);
        }
    }

    public class StackPoolingNode : StackNode
    {
        public int Method { get; set; }

        public override void Count()
        {
            switch (Method)
            {
                case METHOD_AVERAGE:
                    List<double> values = new List<double>();

                    for (int s = 0; s < Values.Count; s++)
                    {
                        foreach (StackUnit unit in InputUnits)
                            values[s] += unit.Values[s];

                        Values[s] = values[s] / InputUnits.Count;
                    }

                    break;

                case METHOD_MAX:
                    for (int s = 0; s < Values.Count; s++)
                    {
                        double max = InputUnits[0].Values[s];

                        for (int i = 1; i < InputUnits.Count; i++)
                            if (InputUnits[i].Values[s] > max)
                                max = InputUnits[i].Values[s];

                        Values[s] = max;
                    }

                    break;

                case METHOD_MIN:
                    for (int s = 0; s < Values.Count; s++)
                    {
                        double min = InputUnits[0].Values[s];

                        for (int i = 1; i < InputUnits.Count; i++)
                            if (InputUnits[i].Values[s] < min)
                                min = InputUnits[i].Values[s];

                        Values[s] = min;
                    }

                    break;
            }
        }

        public const int METHOD_AVERAGE = 0;
        public const int METHOD_MAX = 1;
        public const int METHOD_MIN = 2;

        public StackPoolingNode(List<StackUnit> inputUnits, int method) : this(inputUnits, method, 0) { }
        public StackPoolingNode(List<StackUnit> inputUnits, int method, int stackLength) : base(stackLength)
        {
            Method = method;

            InputUnits = inputUnits;

            foreach (StackUnit unit in inputUnits)
                unit.OutputUnits.Add(this);
        }
    }

    public class StackNeuron : StackUnit
    {
        public override void Count()
        {
            for (int s = 0; s < Values.Count; s++)
            {
                double weightedSum = 0;

                for (int i = 0; i < InputUnits.Count; i++)
                    weightedSum += InputUnits[i].Values[s] * Weights[s][i];

                Values[s] = Tools.Sigmoid(weightedSum + Bias);
            }
        }

        public override void CountDerivatives()
        {
            for (int s = 0; s < Values.Count; s++)
            {
                //double derivative = 0;

                //for (int i = 0; i < OutputUnits.Count; i++)
                //    derivative += OutputUnits[i].Derivatives[s] * Tools.SigmoidDerivative(OutputUnits[i].Values[s]) * OutputUnits[i].Weights[s][i];

                //Derivatives[s] = derivative;

                for (int i = 0; i < InputUnits.Count; i++)
                    InputUnits[i].Derivatives[s] += Derivatives[s] * Tools.SigmoidDerivative(Values[s]) * Weights[s][i];
            }
        }

        public override void ApplyDerivativesToWeights(double learningFactor)
        {
            for (int s = 0; s < Values.Count; s++)
                for (int i = 0; i < Weights.Count; i++)
                    Weights[s][i] += Derivatives[s] * InputUnits[i].Values[s] * Tools.SigmoidDerivative(Values[s]) * learningFactor;
        }

        public override void ApplyDerivativesToBias(double learningFactor)
        {
            for (int s = 0; s < Values.Count; s++)
                Bias += Derivatives[s] * Tools.SigmoidDerivative(Values[s]) * learningFactor;
        }

        public StackNeuron() : this(new List<StackUnit>(), 0) { }
        public StackNeuron(List<StackUnit> inputUnits) : this(inputUnits, 0) { }
        public StackNeuron(int stackLength) : this(new List<StackUnit>(), stackLength) { }
        public StackNeuron(List<StackUnit> inputUnits, int stackLength)
        {
            Values = new List<double>();
            Weights = new List<List<double>>();
            Derivatives = new List<double>();
            Bias = Constants.defaultBias;

            InputUnits = inputUnits;
            OutputUnits = new List<StackUnit>();

            for (int i = 0; i < stackLength; i++)
                Values.Add(0);

            for (int i = 0; i < stackLength; i++)
            {
                Weights.Add(new List<double>());
                for (int j = 0; j < inputUnits.Count; j++)
                    Weights[i].Add(1);
            }

            for (int i = 0; i < stackLength; i++)
                Derivatives.Add(0);

            foreach (StackUnit unit in inputUnits)
                unit.OutputUnits.Add(this);
        }
    }

    public class StackSharedNeuron : StackNeuron
    {
        public List<List<double>> SharedWeights { get; set; }
        public List<int> Indexes { get; set; }

        public override void Count()
        {
            for (int s = 0; s < Values.Count; s++)
            {
                double weightedSum = 0;

                for (int i = 0; i < InputUnits.Count; i++)
                    weightedSum += InputUnits[i].Values[s] * SharedWeights[s][Indexes[i]];

                Values[s] = Tools.Sigmoid(weightedSum + Bias);
            }
        }

        public override void CountDerivatives()
        {
            //FetchWeights();
            //base.CountDerivatives();

            for (int s = 0; s < Values.Count; s++)
                for (int i = 0; i < InputUnits.Count; i++)
                    InputUnits[i].Derivatives[s] += Derivatives[s] * Tools.SigmoidDerivative(Values[s]) * SharedWeights[s][Indexes[i]];
        }

        public override void ApplyDerivativesToWeights(double learningFactor)
        {
            for (int s = 0; s < Values.Count; s++)
                for (int i = 0; i < Weights.Count; i++)
                    SharedWeights[s][Indexes[i]] += Derivatives[s] * InputUnits[i].Values[s] * Tools.SigmoidDerivative(Values[s]) * learningFactor;
        }

        private void FetchWeights()
        {
            for (int s = 0; s < Weights.Count; s++)
                for (int i = 0; i < Weights[s].Count; i++)
                    Weights[s][i] = SharedWeights[s][Indexes[i]];
        }

        public StackSharedNeuron(List<StackUnit> inputUnits, List<List<double>> sharedWeights, List<int> indexes) : this(inputUnits, sharedWeights, indexes, 0) { }
        public StackSharedNeuron(List<StackUnit> inputUnits, List<List<double>> sharedWeights, List<int> indexes, int stackLength) : base(inputUnits, stackLength)
        {
            SharedWeights = sharedWeights;
            Indexes = indexes;
        }
    }

    public class ConvolutionalNeuron : StackNeuron
    {
        public List<List<Filter>> Filters { get; set; }
        public List<int> Indexes { get; set; }

        public override void Count()
        {
            for (int s = 0; s < Values.Count; s++)
            {
                double weightedSum = 0;

                for (int i = 0; i < InputUnits.Count; i++)
                    for (int j = 0; j < InputUnits[i].Values.Count; j++)
                        weightedSum += InputUnits[i].Values[j] * Filters[s][j].Values[Indexes[i]];

                Values[s] = Tools.Sigmoid(weightedSum + Bias);
            }
        }

        public override void CountDerivatives()
        {
            for (int f = 0; f < Values.Count; f++)
            {
                //double derivative = 0;

                //for (int i = 0; i < InputUnits.Count; i++)
                //    for (int j = 0; j < InputUnits[i].Values.Count; j++)
                //        derivative += InputUnits[i].Derivatives[s] * Tools.SigmoidDerivative(OutputUnits[i].Values[s]) * Filters[s][j].Values[Indexes[i]];

                //Derivatives[s] = derivative;

                for (int i = 0; i < InputUnits.Count; i++)
                    for (int s = 0; s < InputUnits[i].Values.Count; s++)
                        InputUnits[i].Derivatives[s] += Derivatives[f] * Tools.SigmoidDerivative(Values[f]) * Filters[f][s].Values[Indexes[i]];
            }
        }

        public override void ApplyDerivativesToWeights(double learningFactor)
        {
            for (int s = 0; s < Values.Count; s++)
                for (int i = 0; i < Weights.Count; i++)
                    for (int j = 0; j < InputUnits[i].Values.Count; j++)
                        Filters[s][j].Values[Indexes[i]] += Derivatives[s] * InputUnits[i].Values[s] * Tools.SigmoidDerivative(Values[s]) * learningFactor;
        }

        public ConvolutionalNeuron(List<StackUnit> inputUnits, List<List<Filter>> filters, List<int> indexes) : this(inputUnits, filters, indexes, 0) { }
        public ConvolutionalNeuron(List<StackUnit> inputUnits, List<List<Filter>> filters, List<int> indexes, int stackLength) : base(inputUnits, stackLength)
        {
            Filters = filters;
            Indexes = indexes;
        }
    }

    public interface IUnit
    {
        void Count();
        void CountDerivatives();
        void ApplyDerivativesToWeights(double learningFactor);
        void ApplyDerivativesToBias(double learningFactor);
    }
}