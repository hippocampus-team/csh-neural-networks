using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;

namespace Testing {
public static class ExperimentsManager {
	private static List<ExperimentLog> experiments = new List<ExperimentLog>();

	public static void loadFromSaved() {
		DirectoryInfo experimentsDirectory = new DirectoryInfo("C://nnlib/experiments");
		if (!experimentsDirectory.Exists) experimentsDirectory.Create();
		else {
			DirectoryInfo[] directories = experimentsDirectory.GetDirectories();
			foreach (DirectoryInfo directory in directories) {
				try {
					string info = File.ReadAllText(directory.FullName + "/info.json");
					if (info.Equals("")) return;
					ExperimentLog experiment = JsonConvert.DeserializeObject<ExperimentLog>(info);
					experiments.Add(experiment);
				} catch (Exception e) {
					Console.WriteLine("Failed to get experiment from {0}", directory.FullName);
				}
			}
		}
	}
	
	public static List<ExperimentMeta> getExperiments() {
		return experiments.Select(experiment => new ExperimentMeta {
			id = experiment.id,
			name = experiment.title,
			description = experiment.description
		}).ToList();
	}
	
	// public static ExperimentMeta? getExperiment(string id) {
	// 	
	// }
}

public struct ExperimentMeta {
	public string id;
	public string name;
	public string description;
}

public struct Experiment {
	public string id;
	public string name;
	public string description;
	public long duration;
	public long initDateTime;
	public long finishDateTime;

	public List<ExperimentPhase> phases;
	public List<LayerTopologyRecord> topologyLayerRecords;
}

public struct ExperimentUpdatePackage {
	public string? name;
	public string? description;
	public List<LayerTopologyRecord>? topologyLayerRecords;
}

public struct ExperimentPhase {
	public string title;
	public string configuration;

	public int iterations;
	public int amountOfParallelNetworks;
	
	public long duration;
	public long startDateTime;
	public long finishDateTime;
}

public struct LayerTopologyRecord {
	public LayerType type;
	public int numberOfUnits;
	public string configuration;
}

public enum LayerType {
	simple,
	dense,
	convolutional,
	pooling
}
}