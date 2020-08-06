using System;
using System.Collections.Generic;
using NeuralNetworks.Units;

namespace NeuralNetworks.Misc {
	public class MatrixModel {
		public int size { get; }
		public int stride { get; }

		public MatrixModel(int size, int stride) {
			this.size = size;
			this.stride = stride;
		}

		public MatrixModel(EList<Unit> layer, int stride) : this(layer.columns, stride) { }

		public int Count() => size * size;

		public int FilterOutputsCount(Filter filter) => (int) Math.Pow((size - filter.size) / stride + 1, 2);

		public int FilterLineCount(Filter filter) => (size - filter.size) / stride + 1;
	}

	public class Filter : ICloneable {
		public int size { get; }
		public List<double> values { get; }

		public Filter(int size) {
			this.size = size;
			values = new List<double>();

			for (int i = 0; i < size * size; i++)
				values.Add(0);
		}

		public Filter(List<double> values) {
			size = (int) Math.Sqrt(values.Count);
			this.values = values;
		}

		public object Clone() => new Filter(values);

		public int Count() => size * size;
	}
}