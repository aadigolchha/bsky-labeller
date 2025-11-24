import express from "express";
import http from "http";
import dotenv from "dotenv";

// --- NUCLEAR FIX START ---
import WebSocket from "ws";
// Force WebSocket to exist globally so Jetstream finds it automatically
// @ts-ignore
global.WebSocket = WebSocket;
// --- NUCLEAR FIX END ---

import { startJetstream } from "./jetstream.js";
import { LabelerServer } from "./labeler-server.js";

dotenv.config();

const app = express();

// --- NEW DID:WEB ROUTE ---
app.get("/.well-known/did.json", (req, res) => {
    res.json({
      "@context": ["https://www.w3.org/ns/did/v1"],
      "id": "did:web:bluesky-labeler.fly.dev", // <--- UPDATE DOMAIN IF NEEDED
      "verificationMethod": [
        {
          "id": "did:web:bluesky-labeler.fly.dev#atproto",
          "type": "Multikey",
          "controller": "did:web:bluesky-labeler.fly.dev",
          "publicKeyMultibase": "zQ3shnktTZyBJyboTXaV9XddXsBvzwGwvL383BzrzW7yy36AT"
        }
      ],
      "service": [
        {
          "id": "#atproto_labeler",
          "type": "AtprotoLabeler",
          "serviceEndpoint": "wss://bluesky-labeler.fly.dev" // <--- UPDATE DOMAIN
        }
      ]
    });
  });
  // -------------------------

const server = http.createServer(app);

const PORT = parseInt(process.env.PORT || "8080", 10);

const labeler = new LabelerServer(process.env.LABELER_DID || "");

app.get("/", (req, res) => {
  res.send("Bluesky Labeler Running");
});

server.on("upgrade", (req, socket, head) => labeler.handleUpgrade(req, socket, head));

server.listen(PORT, "0.0.0.0", async () => {
  await labeler.loadKey(process.env.LABELER_PRIVATE_KEY || "");
  console.log(`Server running on port ${PORT}`);
  startJetstream(labeler);
});