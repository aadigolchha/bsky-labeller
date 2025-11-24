import { WebSocketServer, WebSocket } from "ws";
import { Secp256k1Keypair } from "@atproto/crypto";
import * as cbor from "@ipld/dag-cbor";

export class LabelerServer {
  public wss: WebSocketServer;
  public keypair: Secp256k1Keypair | null = null;
  public did: string;

  constructor(did: string) {
    this.did = did;
    this.wss = new WebSocketServer({ noServer: true });
    
    this.wss.on("connection", (ws) => {
      console.log("🔌 Client connected to label stream");
      ws.on("error", (e) => console.error("WS Error:", e));
    });
  }

  // Load keys from Hex or raw string
  async loadKey(rawKey: string) {
    const cleanHex = rawKey.startsWith("0x") ? rawKey.slice(2) : rawKey;
    this.keypair = await Secp256k1Keypair.import(Buffer.from(cleanHex, "hex"));
    console.log("🔑 Labeler Private Key Loaded");
  }

  // Handle the HTTP Upgrade (The "Traffic Cop")
  handleUpgrade(request: any, socket: any, head: any) {
    if (request.url.startsWith("/xrpc/com.atproto.label.subscribeLabels")) {
      this.wss.handleUpgrade(request, socket, head, (ws) => {
        this.wss.emit("connection", ws, request);
      });
    } else {
      socket.destroy();
    }
  }

  // The Main Event: Broadcast
  async emitLabel(uri: string, val: string, cid?: string) {
    if (!this.keypair) throw new Error("Server not started: No Keypair");

    const label = {
      ver: 1,
      src: this.did,
      uri: uri,   // <--- USE THE EXACT URI PASSED IN
      // cid: cid,   // Optional: specific version
      val: val,
      neg: false,
      cts: new Date().toISOString(),
      ...(cid && { cid }),
    };

    // Sign and Encode
    const toSign = cbor.encode(label);
    const sig = await this.keypair.sign(toSign);
    
    const signedLabel = { ...label, sig };
    
    const header = { t: "#labels", op: 1 };
    const message = { seq: Date.now(), labels: [signedLabel] };

    const frame = Buffer.concat([
      Buffer.from(cbor.encode(header)),
      Buffer.from(cbor.encode(message))
    ]);

    // Send to everyone
    let sent = 0;
    for (const client of this.wss.clients) {
      if (client.readyState === WebSocket.OPEN) {
        client.send(frame);
        sent++;
      }
    }
    // Optional: Log only if people are actually listening
    // if (sent > 0) console.log(`📡 Broadcast '${val}' to ${sent} clients`);
  }
}