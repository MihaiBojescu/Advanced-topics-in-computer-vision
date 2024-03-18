import os
import json
import socketserver
import typing as t
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse
from services.business_logic import BusinessLogic



class HTTPRequestHandler(BaseHTTPRequestHandler):
    _business_logic: BusinessLogic
    _port: int
    _cache: t.Dict[str, str]
    _mime_types: t.Dict[str, str]

    def __init__(self, business_logic: BusinessLogic, port: int, *args):
        self._business_logic = business_logic
        self._port = port
        self._cache = {}
        self._mime_types = {"js": "text/javascript", "html": "text/html", "css": "text/css"}

        for filename in os.listdir("./public"):
            with open(f"./public/{filename}", mode="r", encoding="utf-8") as file:
                self._cache[f"/{filename}"] = file.read()

        self._cache['/'] = self._cache["/index.html"]

        super().__init__(*args)


    def do_GET(self):
        url = urlparse(self.path)

        if url.path not in self._cache:
            self.send_response(404)
            self.end_headers()
            return

        mime_type = self._mime_types[url.path.split(".")[-1] if url.path != '/' else "html"]
        file = self._cache[url.path]

        self.send_response(200)
        self.send_header("Content-type", mime_type)
        self.end_headers()
        self.wfile.write(bytes(file, "utf8"))

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        raw_data = self.rfile.read(content_length)

        coordinates = json.loads(raw_data)
        x = coordinates["x"]
        y = coordinates["y"]

        self._business_logic.run(x, y)
        self.send_response(204)
        self.end_headers()



class Web:
    _business_logic: t.Callable[[int, int], None]
    _port: int

    def __init__(self, business_logic: t.Callable[[int, int], None], port: int) -> None:
        self._business_logic = business_logic
        self._port = port

    def run(self):
        def handler(*args):
            return HTTPRequestHandler(self._business_logic, self._port, *args)

        print(f"Started on port {self._port}")
        server = socketserver.TCPServer(("", self._port), handler)
        server.allow_reuse_address = True
        server.allow_reuse_port = True
        server.serve_forever()
