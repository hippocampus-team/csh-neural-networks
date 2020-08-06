using System.Collections;
using System.Collections.Generic;

namespace NeuralNetworks.Misc {
	public class EList<T> : IList<T> {
		private readonly List<T> array;

		public T this[int row, int column] {
			get => array[column * array.Count + row];
			set => array[column * array.Count + row] = value;
		}

		public int rows => array.Count;
		public int columns { get; }

		public T this[int index] {
			get => array[index];
			set => array[index] = value;
		}

		public int Count => array.Count;
		public bool IsReadOnly => false;

		public EList() : this(1) { }

		public EList(int columns) {
			array = new List<T>();
			this.columns = columns;
		}

		public EList(List<T> array) {
			this.array = array;
			columns = 1;
		}

		public EList(List<List<T>> matrix) {
			array = new List<T>();
			columns = matrix.Count;

			foreach (List<T> row in matrix) array.AddRange(row);
		}

		public IEnumerator<T>   GetEnumerator()                       => array.GetEnumerator();
		IEnumerator IEnumerable.GetEnumerator()                       => GetEnumerator();
		public int              IndexOf(T item)                       => array.IndexOf(item);
		public void             Insert(int index, T item)             => array.Insert(index, item);
		public void             RemoveAt(int index)                   => array.RemoveAt(index);
		public void             Add(T item)                           => array.Add(item);
		public void             Clear()                               => array.Clear();
		public bool             Contains(T item)                      => array.Contains(item);
		public void             CopyTo(T[] copyArray, int arrayIndex) => array.CopyTo(copyArray, arrayIndex);
		public bool             Remove(T item)                        => array.Remove(item);

		public List<T> ToList() => array;
	}
}