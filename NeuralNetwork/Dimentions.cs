using System;
using System.Collections.Generic;

namespace NeuralNetworks
{
    public class MatrixModel
    {
        public int Size { get; set; }
        public int Stride { get; set; }

        public int Count()
        {
            return Size * Size;
        }

        public int FilterOutputsCount(Filter filter)
        {
            return (int)Math.Pow((Size - filter.Size) / Stride + 1, 2);
        }

        public int FilterLineCount(Filter filter)
        {
            return (Size - filter.Size) / Stride + 1;
        }

        public MatrixModel(int size, int stride)
        {
            Size = size;
            Stride = stride;
        }

        public MatrixModel(List<StackUnit> layer, int stride) : this((int)Math.Sqrt(layer.Count), stride) { }
    }

    public class Filter : ICloneable
    {
        public int Size { get; set; }
        public List<double> Values { get; set; }

        public int Count()
        {
            return Size * Size;
        }

        public Filter(int size)
        {
            Size = size;
            Values = new List<double>();

            for (int i = 0; i < size * size; i++)
                Values.Add(0);
        }

        public Filter(List<double> values)
        {
            Size = (int)Math.Sqrt(values.Count);
            Values = values;
        }

        public object Clone()
        {
            Filter clone = new Filter(Values);
            return clone;
        }
    }
}
