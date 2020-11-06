using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using NeuralNetworks;
using NeuralNetworks.Layers;
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
	private readonly string title;
	private readonly string description;

	private TimeSpan duration;
	private readonly DateTime initDateTime;
	private DateTime finishDateTime;

	private readonly List<LogPhase> phases;
	private readonly List<LogTopologyLayerRecord> topologyLayerRecords;
	
	public ExperimentLog() {
		initDateTime = DateTime.Now;
		phases = new List<LogPhase>(4);
		topologyLayerRecords = new List<LogTopologyLayerRecord>();
	}

	public ExperimentLog(string title, string description, NeuralNetwork topologyRecordNetwork = null) : this() {
		this.title = title;
		this.description = description;
		
		if (topologyRecordNetwork != null) 
			recordTopologyFromNetwork(topologyRecordNetwork);
	}

	public void end() {
		finishDateTime = DateTime.Now;
		duration = finishDateTime - initDateTime;
	}

	public void startPhase(string title, string configuration, int iterations, int amountOfParallelNetworks) {
		phases.Add(new LogPhase(title, configuration, iterations, amountOfParallelNetworks));
	}

	public void endPhase() {
		phases.Last().end();
	}
	
	public void recordTopologyFromNetwork(NeuralNetwork nn) {
		topologyLayerRecords.Clear();
		
		foreach (Layer layer in nn.layers)
			topologyLayerRecords.Add(new LogTopologyLayerRecord(layer.GetType().Name, layer.neurons.Count()));
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
}

public class LogPhase {
	private readonly string title;
	private readonly string configuration;

	private readonly int iterations;
	private readonly int amountOfParallelNetworks;
	
	private TimeSpan duration;
	private readonly DateTime startDateTime;
	private DateTime finishDateTime;

	public LogPhase(string title, string configuration, int iterations, int amountOfParallelNetworks) {
		this.title = title;
		this.configuration = configuration;
		this.iterations = iterations;
		this.amountOfParallelNetworks = amountOfParallelNetworks;
		
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
		text += $"Parallel networks: {amountOfParallelNetworks}\n";
		text += $"\n";
		text += $"Duration: {duration}\n";
		text += $"Initiated at: {startDateTime}\n";
		text += $"Finished at: {finishDateTime}\n";
		text += $"\n";

		return text;
	}
}

public class LogTopologyLayerRecord {
	private readonly string type;
	private readonly int numberOfNeurons;
	private readonly string configuration;

	public LogTopologyLayerRecord(string type, int numberOfNeurons, string configuration = "") {
		this.type = type;
		this.numberOfNeurons = numberOfNeurons;
		this.configuration = configuration;
	}

	public string getString() {
		return $"{type} {numberOfNeurons} {configuration}\n";
	}
}

}