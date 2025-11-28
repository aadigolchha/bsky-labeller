/**
 * Jetstream - Bluesky Firehose Consumer
 *
 * Connects to Bluesky's Jetstream firehose and processes incoming posts
 * through the Python misinformation analyzer.
 *
 * The analyzer returns dynamically generated labels based on Community Notes clusters.
 *
 * Memory management:
 * - Periodic Jetstream reconnection to clear internal buffers
 * - Request queue limiting to prevent backpressure buildup
 * - Aggressive GC after every batch
 */


import { Jetstream } from "@skyware/jetstream";
import * as db from "./db.js";
import { fetch, Agent } from "undici";
import type { LabelerServer } from "@skyware/labeler";

// Create a fresh agent - will be recreated periodically
function createAgent(): Agent {
  return new Agent({
    connect: {
      socketPath: "/tmp/analyzer.sock",
      timeout: 2000
    },
    connections: 5,           // Minimal connections
    pipelining: 1,
    keepAliveTimeout: 1000,   // Very short keepAlive
    keepAliveMaxTimeout: 2000,
  });
}

let socketAgent = createAgent();

// Stats tracking - reset periodically to prevent memory growth
let stats = {
  processed: 0,
  analyzed: 0,
  labeled: 0,
  errors: 0,
  consecutiveErrors: 0,
  lastError: null as string | null,
  lastErrorTime: null as Date | null,
};

// Memory management constants
const RECONNECT_INTERVAL = 15000;  // Reconnect Jetstream every 15k posts to clear buffers
const GC_INTERVAL = 5000;          // Trigger GC every 5k posts
const MAX_PENDING_REQUESTS = 50;   // Limit concurrent analyzer requests

// Track state
let processedSinceReconnect = 0;
let processedSinceGC = 0;
let pendingRequests = 0;
let currentJetstream: Jetstream | null = null;
let labelerRef: LabelerServer | null = null;

// Error logging threshold - log after this many consecutive errors
const ERROR_LOG_THRESHOLD = 10;

// Force garbage collection if available (--expose-gc flag)
function triggerGC() {
  if (global.gc) {
    global.gc();
  }
}

// Recreate the HTTP agent to clear connection state
function resetAgent() {
  const oldAgent = socketAgent;
  socketAgent = createAgent();
  // Close old agent after a delay to let pending requests finish
  setTimeout(() => {
    oldAgent.close().catch(() => {});
  }, 5000);
}

// Expected response format from the Python analyzer
// Note: Python now does the mapping, so label is already "misinformation" or "ai-generated"
interface AnalyzerResponse {
  label: string | null;        // Simple label: "misinformation" or "ai-generated"
  cluster_label?: string;      // Original cluster label for logging
  score?: number;
  cluster_id?: number;
  theme?: string;
  severity?: string;
  high_confidence?: boolean;
  text?: string;
  error?: string;
}

// Process a single post - extracted to avoid closure memory retention
async function processPost(
  labeler: LabelerServer,
  text: string,
  did: string,
  uri: string,
  cid: string | undefined
): Promise<void> {
  try {
    pendingRequests++;

    const response = await fetch("http://localhost/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, did }),
      dispatcher: socketAgent,
    });

    if (!response.ok) {
      stats.errors++;
      return;
    }

    const result = await response.json() as AnalyzerResponse;
    stats.analyzed++;
    stats.consecutiveErrors = 0;

    if (result.label) {
      stats.labeled++;

      // result.label is already the simple label ("misinformation" or "ai-generated")
      // result.cluster_label contains the original cluster label for logging
      let logMessage = `[MATCH] Labeled ${did.slice(0, 20)}... as "${result.label}"`;
      if (result.cluster_label) {
        logMessage += ` (cluster: ${result.cluster_label})`;
      }
      if (result.score !== undefined) {
        logMessage += ` - ${(result.score * 100).toFixed(1)}% confidence`;
      }
      if (result.high_confidence) {
        logMessage += " ⚠️ HIGH CONFIDENCE";
      }
      console.log(logMessage);

      // Use label directly - Python already mapped it
      await labeler.createLabel({ uri, cid, val: result.label });
      db.logLabel(uri, cid, result.cluster_label || result.label, text, result.score, result.theme, result.cluster_id);
    }

  } catch (e) {
    stats.errors++;
    stats.consecutiveErrors++;
    stats.lastError = e instanceof Error ? e.message : String(e);
    stats.lastErrorTime = new Date();

    if (stats.consecutiveErrors === ERROR_LOG_THRESHOLD) {
      console.error(`⚠️ Python analyzer connection failing! ${stats.consecutiveErrors} consecutive errors.`);
      console.error(`   Last error: ${stats.lastError}`);
    } else if (stats.consecutiveErrors > 0 && stats.consecutiveErrors % 100 === 0) {
      console.error(`⚠️ Still failing: ${stats.consecutiveErrors} consecutive errors. Last: ${stats.lastError}`);
    }
  } finally {
    pendingRequests--;
  }
}

// Create and start a Jetstream instance
function createJetstream(labeler: LabelerServer): Jetstream {
  const jetstream = new Jetstream({
    wantedCollections: ["app.bsky.feed.post"],
    cursor: db.getCursor(),
  });

  jetstream.on("commit", (event) => {
    if (event.commit.operation === "delete") return;
    if (event.commit.collection !== "app.bsky.feed.post") return;

    stats.processed++;
    processedSinceReconnect++;
    processedSinceGC++;

    // Save cursor periodically
    if (stats.processed % 100 === 0) {
      db.setCursor(event.time_us);
    }

    // Trigger GC frequently
    if (processedSinceGC >= GC_INTERVAL) {
      processedSinceGC = 0;
      triggerGC();
    }

    // Check if we need to reconnect to clear Jetstream's internal buffers
    if (processedSinceReconnect >= RECONNECT_INTERVAL) {
      processedSinceReconnect = 0;
      console.log("🔄 Reconnecting Jetstream to clear memory buffers...");

      // Save cursor before reconnect
      db.setCursor(event.time_us);

      // Schedule reconnection after current event
      setImmediate(() => {
        reconnectJetstream(labeler);
      });
      return; // Skip this event, will resume from cursor
    }

    // Backpressure: skip processing if too many pending requests
    if (pendingRequests >= MAX_PENDING_REQUESTS) {
      return; // Drop this event to prevent memory buildup
    }

    const record = event.commit.record as any;
    if (!record?.text || !event.commit.rkey) return;

    const text: string = record.text;
    const did: string = event.did;
    const uri = `at://${did}/${event.commit.collection}/${event.commit.rkey}`;
    const cid: string | undefined = event.commit.cid;

    // Fire and forget - don't await to prevent backpressure
    processPost(labeler, text, did, uri, cid);
  });

  jetstream.on("error", (error) => {
    console.error("Jetstream error:", error);
  });

  return jetstream;
}

// Reconnect Jetstream to clear internal buffers
function reconnectJetstream(labeler: LabelerServer) {
  if (currentJetstream) {
    try {
      currentJetstream.close();
    } catch (e) {
      // Ignore close errors
    }
  }

  // Reset the HTTP agent too
  resetAgent();

  // Trigger GC before creating new instance
  triggerGC();

  // Create fresh Jetstream
  currentJetstream = createJetstream(labeler);
  currentJetstream.start();

  console.log("✅ Jetstream reconnected, memory cleared");
}

export const startJetstream = (labeler: LabelerServer) => {
  labelerRef = labeler;

  // Periodic stats logging
  setInterval(() => {
    if (stats.processed > 0) {
      const labelRate = ((stats.labeled / stats.processed) * 100).toFixed(4);
      const memUsage = process.memoryUsage();
      const heapMB = (memUsage.heapUsed / 1024 / 1024).toFixed(1);
      console.log(
        `📊 Stats: ${stats.processed} processed, ${stats.analyzed} analyzed, ` +
        `${stats.labeled} labeled (${labelRate}%), ${stats.errors} errors | ` +
        `Heap: ${heapMB}MB | Pending: ${pendingRequests}`
      );
      triggerGC();
    }
  }, 60000);

  // Start Jetstream
  currentJetstream = createJetstream(labeler);
  currentJetstream.start();

  console.log("🚀 Jetstream started (using Unix Sockets)");
  console.log(`   Memory management: Reconnect every ${RECONNECT_INTERVAL} posts, GC every ${GC_INTERVAL} posts`);
  console.log("   Waiting for posts matching Community Notes misinformation patterns...");
};

// Export stats for health checks
export const getStats = () => ({ ...stats });
