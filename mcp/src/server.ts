import { Server } from "mcp-sdk";
const server = new Server({ name: "demo-mcp", version: "0.1.0" });

// Recurso de solo lectura
server.resource("time:now", async () => {
  return { mimeType: "application/json", data: { now: new Date().toISOString() } };
});

// Tool de ejemplo
server.tool("echo", { inputSchema: { type: "object", properties: { text: { type: "string" }}, required: ["text"]}},
  async ({ text }) => ({ content: [{ type: "text", text }]}));

server.listen();
console.log("MCP server listening (demo-mcp)");
