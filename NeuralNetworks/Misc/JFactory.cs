using System;
using NeuralNetworks.Units;
using Newtonsoft.Json.Linq;

namespace NeuralNetworks.Misc {

public class JFactory {
	public static Unit constructUnit(JObject unitJson) {
		Unit unit;
			
		try {
			string typeName = unitJson["type"]?.Value<string>()
							  ?? throw new ArgumentException("Unit type not found in " + unitJson.Path);

			Type unitType = Type.GetType("NeuralNetworks.Units." + typeName);
			if (unitType == null) throw new ArgumentException("Wrong unit type of " + unitJson.Path);
			

			unit = (Unit) unitType.GetMethod("getEmpty")?.Invoke(null, null);
			if (unit == null) throw new ArgumentException("Wrong unit type of " + unitJson.Path);
			
			unit.fillFromJObject(unitJson);
		} catch (Exception e) {
			Console.WriteLine("Error while constructing unit: ");
			Console.WriteLine(e);
			throw;
		}

		return unit;
	}
}

}