using System.Net;
using System.Text;

namespace ServerHost {
public static class Requests {
	public static void api(HttpListenerRequest req, HttpListenerResponse res) {
		res.Headers.Add("Access-Control-Allow-Origin", "*");
		res.StatusCode = (int) HttpStatusCode.OK;
		res.ContentType = "application/json";
		res.ContentEncoding = Encoding.UTF8;
		
		byte[] payload = Encoding.UTF8.GetBytes("{ \"version\": 1.0 }");
		res.OutputStream.Write(payload, 0, payload.Length);
	}
	
	public static void notFound(HttpListenerRequest req, HttpListenerResponse res) {
		res.Headers.Add("Access-Control-Allow-Origin", "*");
		res.StatusCode = (int) HttpStatusCode.NotFound;
	}
	
	public static void getExperimentsList(HttpListenerRequest req, HttpListenerResponse res) {
		res.Headers.Add("Access-Control-Allow-Origin", "*");
		res.StatusCode = (int) HttpStatusCode.OK;
		res.ContentType = "application/json";
		res.ContentEncoding = Encoding.UTF8;
		
		byte[] payload = Encoding.UTF8.GetBytes("[{\"id\":0,\"name\":\"First test\",\"description\":\"Some text\"},{\"id\":1,\"name\":\"One more\",\"description\":\"Very important topic\"}]");
		res.OutputStream.Write(payload, 0, payload.Length);
	}
}
}