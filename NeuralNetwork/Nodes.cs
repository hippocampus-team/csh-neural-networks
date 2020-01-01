using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks
{
    public abstract class Unit
    {
        public double Value { get; set; }
        public double Derivative { get; set; }

        public List<Unit> InputUnits { get; set; }
        public List<Unit> OutputUnits { get; set; }

        public List<double> Weights { get; set; }

        public abstract void Count();
    }

    public class Node : Unit
    {
        public Node() : this(0) { }

        public Node(double value)
        {
            Value = value;
            Derivative = 0;

            InputUnits = null;
            OutputUnits = new List<Unit>();

            Weights = null;
        }

        public override void Count() { }
    }

    public class ReferNode : Node
    {
        public ReferNode(Unit _inputUnit)
        {
            Value = _inputUnit.Value;
            Derivative = 0;

            InputUnits = new List<Unit>();
            OutputUnits = new List<Unit>();

            Weights = new List<double>();

            InputUnits.Add(_inputUnit);
            Weights.Add(1);
            _inputUnit.OutputUnits.Add(this);
            _inputUnit.Weights.Add(1);
        }

        public override void Count()
        {
            Value = InputUnits[0].Value * Weights[0];
        }
    }

    public class Neuron : Unit
    {
        public double Bias { get; set; }

        public override void Count()
        {
            double WeightedSum = 0;

            for (int i = 0; i < InputUnits.Count; i++)
                WeightedSum += InputUnits[i].Value * Weights[i];

            Value = Tools.Sigmoid(WeightedSum - Bias);
        }

        public Neuron() : this(new List<Unit>(), Constants.defaultBias, null) { }

        public Neuron(List<Unit> _inputUnits) : this(_inputUnits, Constants.defaultBias, null) { }

        public Neuron(List<Unit> _inputUnits, double _bias) : this(_inputUnits, _bias, null) { }

        public Neuron(double _bias) : this(new List<Unit>(), _bias, null) { }

        public Neuron(List<Unit> _inputUnits, List<double> _weights) : this(_inputUnits, Constants.defaultBias, _weights) { }

        public Neuron(List<Unit> _inputUnits, double _bias, List<double> _weights)
        {
            Value = 0;
            Weights = new List<double>();
            InputUnits = _inputUnits;
            OutputUnits = new List<Unit>();
            Bias = _bias;

            if (_weights is null)
                for (int i = 0; i < _inputUnits.Count; i++)
                    Weights.Add(1);
            else
                Weights = _weights;

            foreach (Unit unit in _inputUnits)
                unit.OutputUnits.Add(this);
        }
    }
}