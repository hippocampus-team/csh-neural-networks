using System;
using System.Collections.Generic;
using NeuralNetworks.Misc;

namespace NeuralNetworks {
public class MatrixModel {
	public MatrixModel(int size, int stride) {
		Size = size;
		Stride = stride;
	}

	public MatrixModel(EList<Unit> layer, int stride) : this(layer.Columns, stride) { }
	public int Size { get; set; }
	public int Stride { get; set; }

	public int Count() => Size * Size;

	public int FilterOutputsCount(Filter filter) => (int) Math.Pow((Size - filter.Size) / Stride + 1, 2);

	public int FilterLineCount(Filter filter) => (Size - filter.Size) / Stride + 1;
}

public class Filter : ICloneable {
	public Filter(int size) {
		Size = size;
		Values = new List<double>();

		for (int i = 0; i < size * size; i++)
			Values.Add(0);
	}

	public Filter(List<double> values) {
		Size = (int) Math.Sqrt(values.Count);
		Values = values;
	}

	public int Size { get; set; }
	public List<double> Values { get; set; }

	public object Clone() => new Filter(Values);

	public int Count() => Size * Size;
}
}