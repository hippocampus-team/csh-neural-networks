namespace ServerHost {
internal static class Starter {
	// ReSharper disable once InconsistentNaming
	public static void Main(string[] args) {
		int port = 8029;
		
		for (int i = 0; i < args.Length; i++) {
			if (args[i].Equals("--port")) port = int.Parse(args[i + 1]);
		}
		
		HttpServer server = new HttpServer();
		server.start(port);
	}
}
}