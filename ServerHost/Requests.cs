using System.Net;
using System.Text;
using Newtonsoft.Json.Linq;
using Testing;

namespace ServerHost {
public static class Requests {
	public static void api(HttpListenerRequest req, HttpListenerResponse res) {
		setDefaultHeaders(res);
		setHeadersForJson(res);
		
		byte[] payload = Encoding.UTF8.GetBytes("{ \"version\": 1.0 }");
		res.OutputStream.Write(payload, 0, payload.Length);
	}
	
	public static void notFound(HttpListenerRequest req, HttpListenerResponse res) {
		setDefaultHeaders(res);
		res.StatusCode = (int) HttpStatusCode.NotFound;
	}
	
	public static void getExperimentsList(HttpListenerRequest req, HttpListenerResponse res) {
		setDefaultHeaders(res);
		setHeadersForJson(res);
		
		JArray json = JArray.FromObject(ExperimentsManager.getExperiments());
		byte[] payload = Encoding.UTF8.GetBytes(json.ToString());
		res.OutputStream.Write(payload, 0, payload.Length);
	}
	
	public static void getExperiment(HttpListenerRequest req, HttpListenerResponse res) {
		setDefaultHeaders(res);
		setHeadersForJson(res);
		
		
	}

	private static void setDefaultHeaders(HttpListenerResponse res) {
		res.Headers.Add("Access-Control-Allow-Origin", "*");
		res.StatusCode = (int) HttpStatusCode.OK;
	}
	
	private static void setHeadersForJson(HttpListenerResponse res) {
		res.ContentType = "application/json";
		res.ContentEncoding = Encoding.UTF8;
	}
}
}