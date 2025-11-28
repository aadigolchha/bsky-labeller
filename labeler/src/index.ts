import dotenv from "dotenv";
dotenv.config();

// --- NUCLEAR FIX START ---
import WebSocket from "ws";
// Force WebSocket to exist globally so Jetstream finds it automatically
// @ts-ignore
global.WebSocket = WebSocket;
// --- NUCLEAR FIX END ---

import { LabelerServer } from "@skyware/labeler";
import { startJetstream } from "./jetstream.js";

// Validate required environment variables
const LABELER_DID = process.env.LABELER_DID;
const LABELER_PRIVATE_KEY = process.env.LABELER_PRIVATE_KEY;

console.log("🔧 Starting Bluesky Labeler...");
console.log(`   DID: ${LABELER_DID ? LABELER_DID.slice(0, 20) + '...' : 'NOT SET'}`);
console.log(`   Key: ${LABELER_PRIVATE_KEY ? '***SET***' : 'NOT SET'}`);
console.log(`   NODE_ENV: ${process.env.NODE_ENV}`);

if (!LABELER_DID || !LABELER_DID.startsWith('did:')) {
  console.error("❌ FATAL: LABELER_DID must be set and start with 'did:' in .env");
  process.exit(1);
}

if (!LABELER_PRIVATE_KEY) {
  console.error("❌ FATAL: LABELER_PRIVATE_KEY must be set in .env");
  process.exit(1);
}

const PORT = parseInt(process.env.PORT || "8080", 10);
const dbPath = process.env.NODE_ENV === 'production' ? '/data/labels.db' : './labels.db';

console.log(`   Port: ${PORT}`);
console.log(`   DB Path: ${dbPath}`);

// Use @skyware/labeler which properly handles label persistence and emission
let server: LabelerServer;
try {
  console.log("🔧 Creating LabelerServer instance...");
  server = new LabelerServer({
    did: LABELER_DID,
    signingKey: LABELER_PRIVATE_KEY,
    dbPath: dbPath,
  });
  console.log("✅ LabelerServer instance created");
} catch (err) {
  console.error("❌ FATAL: Failed to create LabelerServer:", err);
  process.exit(1);
}

// Start the labeler server
console.log(`🔧 Calling server.start() on port ${PORT}...`);
server.start({ port: PORT, host: "0.0.0.0" }, (error, address) => {
  if (error) {
    console.error("❌ Failed to start labeler server:", error);
    process.exit(1);
  }
  console.log(`🚀 Labeler server listening on ${address}`);

  // Start the Jetstream firehose consumer
  startJetstream(server);
});

// Handle uncaught errors
process.on('uncaughtException', (err) => {
  console.error('❌ Uncaught Exception:', err);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('❌ Unhandled Rejection at:', promise, 'reason:', reason);
});
