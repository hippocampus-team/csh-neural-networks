using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;

namespace ServerHost {
public class RequestsMapper {
	private const string notFoundRequest = "notFound";

	private readonly Dictionary<string, string> pathDictionary;

	public RequestsMapper() {
		pathDictionary = new Dictionary<string, string> {
			{ "GET:/api", "api" }, 
			{ "GET:/experiments", "getExperimentsList" }
		};
	}

	public MethodInfo getRequestMethod(string httpMethod, string path) {
		string key = httpMethod.ToUpper() + ":" + path.ToLower();

		if (!pathDictionary.TryGetValue(key, out string? methodSignature))
			methodSignature = notFoundRequest;
		
		MethodInfo? method = typeof(Requests).GetMethod(methodSignature);
		Debug.Assert(method != null, nameof(method) + " != null");
		return method;
	}
}
}