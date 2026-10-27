"use strict";

// Dependency-free server for the self-contained showcase. It intentionally exposes only this
// directory, accepts GET/HEAD, and has no API or directory listing.
const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");

const root = __dirname;
const port = Number.parseInt(process.env.PORT || "8080", 10);
const types = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
};

function send(response, status, headers, body) {
  response.writeHead(status, headers);
  response.end(body);
}

http.createServer((request, response) => {
  if (request.method !== "GET" && request.method !== "HEAD") {
    return send(response, 405, { Allow: "GET, HEAD" });
  }

  const pathname = new URL(request.url, `http://${request.headers.host || "localhost"}`).pathname;
  const relative = pathname === "/" ? "index.html" : decodeURIComponent(pathname).replace(/^\/+/, "");
  const file = path.resolve(root, relative);
  if (file !== root && !file.startsWith(`${root}${path.sep}`)) {
    return send(response, 403, { "Content-Type": "text/plain; charset=utf-8" }, "Forbidden");
  }

  fs.stat(file, (error, stat) => {
    if (error || !stat.isFile()) {
      return send(response, 404, { "Content-Type": "text/plain; charset=utf-8" }, "Not found");
    }
    response.writeHead(200, {
      "Content-Type": types[path.extname(file).toLowerCase()] || "application/octet-stream",
      "Content-Length": stat.size,
      "X-Content-Type-Options": "nosniff",
      "Referrer-Policy": "no-referrer",
    });
    if (request.method === "HEAD") return response.end();
    fs.createReadStream(file).pipe(response);
  });
}).listen(port, "127.0.0.1", () => {
  console.log(`FedBraTS showcase listening on http://127.0.0.1:${port}`);
});
