/**
 * Caller Listener
 *
 * HTTP endpoint that receives report events (POST /caller/report).
 * Accepts report payloads, validates input, passes to Analyzer then Reactor,
 * and returns status + label applied to the client.
 *
 * Use startCallerServer(labeler, port) to run a standalone HTTP server, or
 * createReportHandler(labeler) for Fastify integration.
 */

import { createServer, type IncomingMessage, type ServerResponse } from "http";
import type { FastifyRequest, FastifyReply } from "fastify";
import type { LabelerServer } from "@skyware/labeler";
import { analyzeReport } from "./caller-analyzer.js";
import { executeLabelAction } from "./caller-reactor.js";
import type { ReportPayload } from "./caller-analyzer.js";

const LOG_PREFIX = "[CALLER-LISTENER]";

/** Body expected from POST /caller/report */
type ReportBody = {
  uri?: string;
  cid?: string;
  reason?: string;
};

function validatePayload(body: unknown): { ok: true; payload: ReportPayload } | { ok: false; message: string } {
  if (body == null || typeof body !== "object" || Array.isArray(body)) {
    return { ok: false, message: "Request body must be a JSON object" };
  }

  const b = body as Record<string, unknown>;
  const uri = typeof b.uri === "string" ? b.uri.trim() : "";

  if (!uri) {
    return { ok: false, message: "Missing or empty required field: uri" };
  }

  if (!uri.startsWith("at://")) {
    return { ok: false, message: "uri must be an AT Protocol URI (at://...)" };
  }

  const payload: ReportPayload = {
    uri,
    cid: typeof b.cid === "string" ? b.cid.trim() || undefined : undefined,
    reason: typeof b.reason === "string" ? b.reason.trim() || undefined : undefined,
  };

  return { ok: true, payload };
}

async function handleReport(
  body: unknown,
  labeler: LabelerServer
): Promise<{ statusCode: number; body: object }> {
  const validation = validatePayload(body);
  if (!validation.ok) {
    console.log(`${LOG_PREFIX} Validation failed: ${validation.message}`);
    return { statusCode: 400, body: { status: "error", error: validation.message } };
  }

  const { payload } = validation;
  console.log(`${LOG_PREFIX} Valid payload: uri=${payload.uri} cid=${payload.cid ?? "(none)"} reason=${payload.reason ?? "(none)"}`);
  const { label } = await analyzeReport(payload);
  console.log(`${LOG_PREFIX} Analyzer chose label: ${label}`);
  const result = await executeLabelAction(payload, label, labeler);

  if (result.success) {
    return { statusCode: 200, body: { status: "success", label: result.label } };
  }
  return {
    statusCode: 500,
    body: { status: "error", error: result.error ?? "Failed to apply label" },
  };
}

/**
 * Returns a Fastify route handler for POST /caller/report.
 * Register with: server.app.post("/caller/report", createReportHandler(server));
 */
export function createReportHandler(labeler: LabelerServer) {
  return async function reportHandler(
    request: FastifyRequest<{ Body: ReportBody }>,
    reply: FastifyReply
  ): Promise<void> {
    console.log(`${LOG_PREFIX} POST /caller/report received`);
    try {
      const { statusCode, body } = await handleReport(request.body, labeler);
      await reply.status(statusCode).send(body);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(`${LOG_PREFIX} Unexpected error:`, err);
      await reply.status(500).send({ status: "error", error: message });
    }
  };
}

/**
 * Starts a standalone HTTP server for the caller report endpoint.
 * Bypasses Fastify to avoid integration issues with @skyware/labeler.
 * Listens on CALLER_PORT (default: 8081).
 */
export function startCallerServer(
  labeler: LabelerServer,
  port: number
): ReturnType<typeof createServer> {
  const server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
    if (req.method !== "POST" || req.url !== "/caller/report") {
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "error", error: "Not Found" }));
      return;
    }

    let body: unknown;
    try {
      const raw = await new Promise<string>((resolve, reject) => {
        const chunks: Buffer[] = [];
        req.on("data", (chunk) => chunks.push(chunk as Buffer));
        req.on("end", () => resolve(Buffer.concat(chunks).toString("utf-8")));
        req.on("error", reject);
      });
      body = JSON.parse(raw || "{}");
    } catch {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "error", error: "Invalid JSON body" }));
      return;
    }

    console.log(`${LOG_PREFIX} POST /caller/report received`);
    try {
      const { statusCode, body: responseBody } = await handleReport(body, labeler);
      res.writeHead(statusCode, { "Content-Type": "application/json" });
      res.end(JSON.stringify(responseBody));
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(`${LOG_PREFIX} Unexpected error:`, err);
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "error", error: message }));
    }
  });

  server.listen(port, "0.0.0.0", () => {
    console.log(`✅ Caller report server listening on http://0.0.0.0:${port}/caller/report`);
  });

  return server;
}
