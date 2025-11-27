/**
 * Jetstream - Bluesky Firehose Consumer
 * 
 * Connects to Bluesky's Jetstream firehose and processes incoming posts
 * through the Python misinformation analyzer.
 * 
 * The analyzer returns dynamically generated labels based on Community Notes clusters.
 */

import { Jetstream } from "@skyware/jetstream";
import * as db from "./db.js";
import { fetch, Agent } from "undici";
import { LabelerServer } from "./labeler-server.js";

// Create a specialized Agent for Unix Domain Socket communication
const socketAgent = new Agent({
  connect: {
    socketPath: "/tmp/analyzer.sock",
    timeout: 2000  // 2 second timeout
  },
  connections: 20,
  pipelining: 1,
});

// Stats tracking
let stats = {
  processed: 0,
  analyzed: 0,
  labeled: 0,
  errors: 0,
  consecutiveErrors: 0,
  lastError: null as string | null,
  lastErrorTime: null as Date | null,
};

// Error logging threshold - log after this many consecutive errors
const ERROR_LOG_THRESHOLD = 10;

// Expected response format from the Python analyzer
interface AnalyzerResponse {
  label: string | null;
  score?: number;
  cluster_id?: number;
  theme?: string;
  severity?: string;
  high_confidence?: boolean;
  text?: string;
  error?: string;
}

export const startJetstream = (labeler: LabelerServer) => {
  const jetstream = new Jetstream({
    wantedCollections: ["app.bsky.feed.post"],
    cursor: db.getCursor(),
  });

  // Periodic stats logging
  setInterval(() => {
    if (stats.processed > 0) {
      const labelRate = stats.processed > 0 
        ? ((stats.labeled / stats.processed) * 100).toFixed(4) 
        : "0";
      console.log(
        `📊 Stats: ${stats.processed} processed, ${stats.analyzed} analyzed, ` +
        `${stats.labeled} labeled (${labelRate}%), ${stats.errors} errors`
      );
    }
  }, 60000);  // Every minute

  jetstream.on("commit", async (event) => {
    // Skip deletions
    if (event.commit.operation === "delete") return;
    
    // Only process posts
    if (event.commit.collection !== "app.bsky.feed.post") return;

    stats.processed++;

    // Periodically save cursor (every ~100 events)
    if (event.time_us % 100 === 0) {
      db.setCursor(event.time_us);
    }

    // Extract post content
    const record = event.commit.record as any;
    if (!record || !record.text) return;

    const text: string = record.text;
    const did: string = event.did;

    // Validate required fields
    if (!event.commit.rkey) {
      stats.errors++;
      return;
    }

    const uri = `at://${did}/${event.commit.collection}/${event.commit.rkey}`;
    const cid: string | undefined = event.commit.cid;

    try {
      // Send to Python analyzer via Unix socket
      const response = await fetch("http://localhost/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, did }),
        dispatcher: socketAgent,
      });

      if (!response.ok) {
        // Python analyzer returned an error status
        stats.errors++;
        return;
      }

      const result = await response.json() as AnalyzerResponse;
      stats.analyzed++;
      stats.consecutiveErrors = 0; // Reset on success

      // Check if a label was returned
      if (result.label) {
        stats.labeled++;

        // Build log message with available context
        let logMessage = `[MATCH] Labeled ${did.slice(0, 20)}... as "${result.label}"`;
        
        if (result.score !== undefined) {
          logMessage += ` (${(result.score * 100).toFixed(1)}% confidence)`;
        }
        
        if (result.theme) {
          logMessage += ` - Theme: ${result.theme}`;
        }
        
        if (result.high_confidence) {
          logMessage += " ⚠️ HIGH CONFIDENCE";
        }

        console.log(logMessage);

        // Emit the label to connected subscribers
        labeler.emitLabel(uri, result.label, cid);

        // Log to database with extended info
        db.logLabel(
          uri, 
          cid, 
          result.label, 
          text,
          result.score,
          result.theme,
          result.cluster_id
        );
      }

    } catch (e) {
      stats.errors++;
      stats.consecutiveErrors++;
      stats.lastError = e instanceof Error ? e.message : String(e);
      stats.lastErrorTime = new Date();

      // Log errors after threshold to avoid flooding, but ensure visibility
      if (stats.consecutiveErrors === ERROR_LOG_THRESHOLD) {
        console.error(`⚠️ Python analyzer connection failing! ${stats.consecutiveErrors} consecutive errors.`);
        console.error(`   Last error: ${stats.lastError}`);
        console.error(`   Is the Python analyzer running? Check /tmp/analyzer.sock`);
      } else if (stats.consecutiveErrors > 0 && stats.consecutiveErrors % 100 === 0) {
        // Periodic reminder for persistent issues
        console.error(`⚠️ Still failing: ${stats.consecutiveErrors} consecutive errors. Last: ${stats.lastError}`);
      }
    }
  });

  jetstream.on("error", (error) => {
    console.error("Jetstream error:", error);
  });

  jetstream.start();
  console.log("🚀 Jetstream started (using Unix Sockets)");
  console.log("   Waiting for posts matching Community Notes misinformation patterns...");
};

// Export stats for health checks
export const getStats = () => ({ ...stats });
