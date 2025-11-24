import express from "express";
import http from "http";
import dotenv from "dotenv";
import { startJetstream } from "./jetstream.js";
import { LabelerServer } from "./labeler-server.js";

dotenv.config();

const app = express();
const server = http.createServer(app);
const PORT = process.env.PORT || 8080;

// Initialize the Class
// Ensure LABELER_DID is set in .env
const labeler = new LabelerServer(process.env.LABELER_DID || "");

app.get("/", (req, res) => res.send("Bluesky Labeler Running"));

// Hook up the upgrade handler
server.on("upgrade", (req, socket, head) => labeler.handleUpgrade(req, socket, head));

server.listen(PORT, async () => {
  await labeler.loadKey(process.env.LABELER_PRIVATE_KEY || "");
  console.log(`Server running on port ${PORT}`);
  
  // Pass the labeler instance to jetstream so it can broadcast
  startJetstream(labeler);
});