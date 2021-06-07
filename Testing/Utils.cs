using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using NeuralNetworks;
using NeuralNetworks.Layers;
using Newtonsoft.Json.Linq;
using OfficeOpenXml;
using OfficeOpenXml.Style;

namespace Testing {

public class Utils {
	public static int getNextByte(Stream fileStream) {
		byte[] digit = new byte[1];
		fileStream.Read(digit, 0, 1);
		return digit[0];
	}

	public static void writeToExcel<T>(NNsTestResults<T> testResults, string rootPath) {
		createRootResultsDirectoryIfNotExists(rootPath);

		List<int> correctAnswersAmount = testResults.correctAnswersAmount;
		List<T> correctAnswers = testResults.correctAnswers;
		List<List<T>> answers = testResults.answers;

		ExcelPackage.LicenseContext = LicenseContext.NonCommercial;
		using ExcelPackage package = new ExcelPackage(new FileInfo($"{rootPath}/test_results.xlsx"));
		ExcelWorksheet worksheet = package.Workbook.Worksheets.Add("Results");

		worksheet.Cells[1, 1].Value = "EXP";
		worksheet.Cells[1, 1].Style.Font.Bold = true;
		worksheet.Cells[2, 1].Value = correctAnswers.Count;
		worksheet.Column(1).Style.Border.Right.Style = ExcelBorderStyle.Thin;

		for (int i = 0; i < correctAnswers.Count; i++)
			worksheet.Cells[i + 4, 1].Value = correctAnswers[i];

		for (int nnIndex = 0; nnIndex < answers.Count; nnIndex++) {
			int currentColumn = nnIndex + 2;

			worksheet.Cells[1, currentColumn].Value = "NN " + (nnIndex + 1);
			worksheet.Cells[1, currentColumn].Style.Font.Bold = true;

			worksheet.Cells[2, currentColumn].Value = correctAnswersAmount[nnIndex];
			worksheet.Cells[3, currentColumn].Value = correctAnswersAmount[nnIndex] * 1d / answers[nnIndex].Count;

			for (int answerIndex = 0; answerIndex < answers[nnIndex].Count; answerIndex++) {
				worksheet.Cells[4 + answerIndex, currentColumn].Value = answers[nnIndex][answerIndex];
				if (!answers[nnIndex][answerIndex].Equals(correctAnswers[answerIndex])) 
					worksheet.Cells[4 + answerIndex, currentColumn].Style.Font.Color.SetColor(Color.Red);
			}

			worksheet.Column(currentColumn).Style.Border.Right.Style = ExcelBorderStyle.Thin;
		}

		package.Save();
	}

	public static void writeNNsToFiles(List<NeuralNetwork> nns, string rootPath) {
		createRootResultsDirectoryIfNotExists(rootPath);

		for (int i = 0; i < nns.Count; i++)
			File.WriteAllText($"{rootPath}/nn_{i}.json", nns[i].serialize());
	}
	
	public static void endLogAndWriteToFile(ExperimentLog log, string rootPath) {
		createRootResultsDirectoryIfNotExists(rootPath);
		
		log.end();
		File.WriteAllText($"{rootPath}/log.txt", log.getString());
		File.WriteAllText($"{rootPath}/info.json", log.getJsonString());
	}

	private static void createRootResultsDirectoryIfNotExists(string rootPath) {
		if (!Directory.Exists(rootPath)) Directory.CreateDirectory(rootPath);
	}
}

public class NNsTestResults <T> {
	public List<int> correctAnswersAmount { get; }
	public List<T> correctAnswers { get; }
	public List<List<T>> answers { get; }

	public NNsTestResults(List<int> correctAnswersAmount, List<T> correctAnswers, List<List<T>> answers) {
		this.correctAnswersAmount = correctAnswersAmount;
		this.correctAnswers = correctAnswers;
		this.answers = answers;
	}
}

public class ExperimentLog {
	public string id;
	public string title;
	public string description;
	
	public int amountOfParallelNetworks;

	public TimeSpan duration;
	public DateTime initDateTime;
	public DateTime finishDateTime;

	public List<LogPhase> phases;
	public List<LogTopologyLayerRecord> topologyLayerRecords;
	
	public ExperimentLog() {
		id = new Random(new Guid().GetHashCode()).Next(Int32.MaxValue - 1).ToString();
		initDateTime = DateTime.Now;
		phases = new List<LogPhase>(4);
		topologyLayerRecords = new List<LogTopologyLayerRecord>();
	}
	
	public ExperimentLog(int amountOfParallelNetworks) : this() {
		this.amountOfParallelNetworks = amountOfParallelNetworks;
	}

	public ExperimentLog(string title, string description, int amountOfParallelNetworks, 
						 NeuralNetwork topologyRecordNetwork = null) : this(amountOfParallelNetworks) {
		this.title = title;
		this.description = description;
		this.amountOfParallelNetworks = amountOfParallelNetworks;
		
		if (topologyRecordNetwork != null) 
			recordTopologyFromNetwork(topologyRecordNetwork);
	}

	public void end() {
		finishDateTime = DateTime.Now;
		duration = finishDateTime - initDateTime;
	}

	public void startPhase(string title, string configuration, int iterations) {
		phases.Add(new LogPhase(title, configuration, iterations));
	}

	public void endPhase() {
		phases.Last().end();
	}
	
	public void recordTopologyFromNetwork(NeuralNetwork nn) {
		topologyLayerRecords.Clear();
		
		foreach (Layer layer in nn.layers)
			topologyLayerRecords.Add(new LogTopologyLayerRecord(layer.layerType, layer.units.Count()));
	}

	public string getString() {
		string text = "";

		text += $"Title: {title}\n";
		text += $"Description: {description}\n";
		text += $"\n";
		text += $"Duration: {duration}\n";
		text += $"Initiated at: {initDateTime}\n";
		text += $"Finished at: {finishDateTime}\n";
		text += $"\n";
		
		text += $"Topology record: \n";
		text = topologyLayerRecords.Aggregate(text, (current, topologyLayerRecord) 
												  => current + topologyLayerRecord.getString());
		text += $"\n";
		
		text += $"Phases: \n";
		text = phases.Aggregate(text, (current, phase) 
									=> current + phase.getString());
		text += $"\n";

		return text;
	}
	
	public string getJsonString() {
		return JObject.FromObject(this).ToString();
	}
}

public class LogPhase {
	public string title;
	public string configuration;

	public int iterations;

	public TimeSpan duration;
	public DateTime startDateTime;
	public DateTime finishDateTime;

	public LogPhase(string title, string configuration, int iterations) {
		this.title = title;
		this.configuration = configuration;
		this.iterations = iterations;
		
		startDateTime = DateTime.Now;
	}

	public void end() {
		finishDateTime = DateTime.Now;
		duration = finishDateTime - startDateTime;
	}

	public string getString() {
		string text = "";

		text += $"/// {title}\n";
		text += $"Configuration: {configuration}\n";
		text += $"Iterations: {iterations}\n";
		text += $"\n";
		text += $"Duration: {duration}\n";
		text += $"Initiated at: {startDateTime}\n";
		text += $"Finished at: {finishDateTime}\n";
		text += $"\n";

		return text;
	}
}

public class LogTopologyLayerRecord {
	public LayerType type;
	public int numberOfNeurons;

	public LogTopologyLayerRecord(LayerType type, int numberOfNeurons) {
		this.type = type;
		this.numberOfNeurons = numberOfNeurons;
	}

	public string getString() {
		return $"{type} {numberOfNeurons}\n";
	}
}

}