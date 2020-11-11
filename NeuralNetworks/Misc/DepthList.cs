using System.Collections;
using System.Collections.Generic;

namespace NeuralNetworks.Misc {

public class DepthList<T> : IList<T> {
	private readonly List<T> array;

	public T this[int index, int depthIndex] {
		get => array[depthIndex * length + index];
		set => array[depthIndex * length + index] = value;
	}

	public int length => array.Count / depth;
	public int depth { get; }

	public DepthList() : this(1) { }

	public DepthList(int depth) {
		array = new List<T>();
		this.depth = depth;
	}

	public DepthList(List<T> array) {
		this.array = array;
		depth = 1;
	}

	public DepthList(List<List<T>> matrix) {
		array = new List<T>();
		depth = matrix.Count;

		foreach (List<T> arr in matrix) array.AddRange(arr);
	}

	public T this[int index] {
		get => array[index];
		set => array[index] = value;
	}

	public List<T> getListOfDepthLevel(int depthLevel) {
		List<T> list = new List<T>(length);

		for (int i = depthLevel * length; i < (depthLevel + 1) * length; i++) 
			list.Add(array[i]);

		return list;
	}

	public int Count => array.Count;
	public bool IsReadOnly => false;

	public IEnumerator<T> GetEnumerator() => array.GetEnumerator();
	IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
	public int IndexOf(T item) => array.IndexOf(item);
	public void Insert(int index, T item) => array.Insert(index, item);
	public void RemoveAt(int index) => array.RemoveAt(index);
	public void Add(T item) => array.Add(item);
	public void Clear() => array.Clear();
	public bool Contains(T item) => array.Contains(item);
	public void CopyTo(T[] copyArray, int arrayIndex) => array.CopyTo(copyArray, arrayIndex);
	public bool Remove(T item) => array.Remove(item);
	public List<T> toList() => array; // Actually asList()
}

}