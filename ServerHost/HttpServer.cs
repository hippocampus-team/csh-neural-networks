using System;
using System.Net;
using System.Reflection;
using System.Threading.Tasks;

namespace ServerHost {
    public class HttpServer {
        private readonly HttpListener listener;
        private readonly RequestsMapper requestsMapper;
        private bool isRunning;
        
        public HttpServer() {
            listener = new HttpListener();
            requestsMapper = new RequestsMapper();
            isRunning = false;
        }

        public void start(int port) {
            listener.Prefixes.Add("http://localhost:" + port + "/");
            listener.Start();
            Console.WriteLine("Listening for connections on http://localhost:{0}/", port);

            // Handle requests
            Task listenTask = handleIncomingConnections();
            listenTask.GetAwaiter().GetResult();
            
            listener.Close();
        }

        private async Task handleIncomingConnections() {
            isRunning = true;
            
            while (isRunning) {
                // Will wait here until we hear from a connection
                HttpListenerContext ctx = await listener.GetContextAsync();
                
                HttpListenerRequest req = ctx.Request;
                HttpListenerResponse res = ctx.Response;
                
                Console.WriteLine("Received {0} request on: {1}", 
                                  req.HttpMethod, req.Url != null ? req.Url.AbsolutePath : "/");

                MethodInfo requestMethod = requestsMapper.getRequestMethod(req.HttpMethod, req.Url != null ? req.Url.AbsolutePath : "/");
                requestMethod.Invoke(null, new object[] { req, res });
                res.Close();
            }
        }
    }
}