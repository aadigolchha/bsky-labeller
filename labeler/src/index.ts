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