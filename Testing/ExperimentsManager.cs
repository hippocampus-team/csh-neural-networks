using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NeuralNetworks.Layers;
using Newtonsoft.Json;

namespace Testing {
public static class ExperimentsManager {
	private static Dictionary<string, ExperimentLog> experiments = new Dictionary<string, ExperimentLog>();

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
					experiments.Add(experiment.id, experiment);
				} catch (Exception e) {
					Console.WriteLine("Failed to get experiment from {0}", directory.FullName);
				}
			}
		}
	}
	
	public static List<ExperimentMeta> getExperiments() {
		return experiments.Select(experiment => new ExperimentMeta {
			id = experiment.Key,
			name = experiment.Value.title,
			description = experiment.Value.description
		}).ToList();
	}
	
	public static Experiment? getExperiment(string id) {
		ExperimentLog log = experiments.First(experiment => experiment.Key == id).Value;
		
		return new Experiment {
			id = log.id,
			name = log.title,
			description = log.description,
			amountOfParallelNetworks = log.amountOfParallelNetworks,
			duration = (long) Math.Round(log.duration.TotalMilliseconds),
			initDateTime = log.initDateTime.Ticks / 1000,
			finishDateTime = log.finishDateTime.Ticks / 1000,
			phases = log.phases.Select(phase => new ExperimentPhase {
				title = phase.title,
				configuration = phase.configuration,
				iterations = phase.iterations,
				duration = (long) Math.Round(phase.duration.TotalMilliseconds),
				startDateTime = phase.startDateTime.Ticks / 1000,
				finishDateTime = phase.finishDateTime.Ticks / 1000,
			}).ToList(),
			topologyLayerRecords = log.topologyLayerRecords.Select(record => new LayerTopologyRecord {
				type = record.type,
				numberOfUnits = record.numberOfNeurons
			}).ToList()
		};
	}
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
	public int amountOfParallelNetworks;
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

	public long duration;
	public long startDateTime;
	public long finishDateTime;
}

public struct LayerTopologyRecord {
	public LayerType type;
	public int numberOfUnits;
	public string configuration;
}
}