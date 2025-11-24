import { Jetstream } from "@skyware/jetstream";
import WebSocket from "ws";
import * as db from "./db.js";
// import { broadcastLabel } from "./labeler-server.js"; 

// 1. Import Undici for Socket Support
import { fetch, Agent } from "undici";
import { LabelerServer } from "./labeler-server.js";

// 2. Create a specialized Agent that looks at the socket file
const socketAgent = new Agent({
  connect: {
    socketPath: "/tmp/analyzer.sock", // Must match the Python --uds flag
    timeout: 1000 // Fail fast if Python is stuck
  },
  connections: 20,
  pipelining: 1,
});

export const startJetstream = (labeler: LabelerServer) => {
  const jetstream = new Jetstream({
    wantedCollections: ["app.bsky.feed.post"],
    cursor: db.getCursor(),
  });

  jetstream.on("commit", async (event) => {
    // Standard filters
    if (event.commit.operation === "delete") return;
    if (event.commit.collection !== "app.bsky.feed.post") return;

    if (event.time_us % 100 === 0) db.setCursor(event.time_us);

    const record = event.commit.record as any;
    if (!record || !record.text) return;

    const text = record.text;
    const did = event.did;
    const uri = `at://${did}/${event.commit.collection}/${event.commit.rkey}`;
    const cid = event.commit.cid;

    try {
      // 3. Use the undici fetch with the socket agent
      // Note: The URL "http://localhost/analyze" is a placeholder. 
      // The dispatcher ignores the domain and sends it straight to the socket.
      const response = await fetch("http://localhost/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, did }),
        dispatcher: socketAgent, // <--- THIS IS THE MAGIC FIX
      });

      // Handle non-200 responses (e.g. Python crashes)
      if (!response.ok) {
        // console.warn("Python busy/error");
        return; 
      }

      const result = await response.json() as any;

      if (result.label) {
        console.log(`[MATCH] Labeled ${did} as ${result.label}`);
        
        labeler.emitLabel(did, result.label, cid);
        db.logLabel(uri, cid, result.label, text);
      }
    } catch (e) {
      // 4. Swallow connection errors quietly so logs aren't flooded
      // "Undici" errors often look like "SocketError: other side closed"
      // console.error("Python unavailable");
    }
  });

  jetstream.start();
  console.log("Jetstream started (using Unix Sockets)...");
};