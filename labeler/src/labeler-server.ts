import { WebSocketServer, WebSocket } from "ws";
import { Secp256k1Keypair } from "@atproto/crypto";
import * as cbor from "@ipld/dag-cbor";

// Stored label type (with signature as $bytes for ATProto JSON serialization)
export interface StoredLabel {
  ver: number;
  src: string;
  uri: string;
  cid?: string;
  val: string;
  neg: boolean;
  cts: string;
  sig: { $bytes: string }; // ATProto $bytes format (base64 encoded)
}

export class LabelerServer {
  public wss: WebSocketServer;
  public keypair: Secp256k1Keypair | null = null;
  public did: string;

  // Store labels in memory, keyed by URI for fast lookup
  private labelsByUri: Map<string, StoredLabel[]> = new Map();
  private allLabels: StoredLabel[] = [];

  constructor(did: string) {
    this.did = did;
    this.wss = new WebSocketServer({ noServer: true });

    this.wss.on("connection", (ws) => {
      console.log("🔌 Client connected to label stream");
      ws.on("error", (e) => console.error("WS Error:", e));
    });
  }

  // Query labels for given URIs (implements com.atproto.label.queryLabels)
  queryLabels(uriPatterns: string[], sources?: string[], cursor?: string, limit: number = 50): { cursor?: string; labels: StoredLabel[] } {
    let results: StoredLabel[] = [];

    // If no URI patterns, return recent labels
    if (!uriPatterns || uriPatterns.length === 0) {
      results = this.allLabels.slice(-limit);
    } else {
      // Match labels by URI patterns
      for (const pattern of uriPatterns) {
        if (pattern.includes('*')) {
          // Wildcard matching
          const regex = new RegExp('^' + pattern.replace(/\*/g, '.*') + '$');
          for (const [uri, labels] of this.labelsByUri) {
            if (regex.test(uri)) {
              results.push(...labels);
            }
          }
        } else {
          // Exact match
          const labels = this.labelsByUri.get(pattern);
          if (labels) {
            results.push(...labels);
          }
        }
      }
    }

    // Filter by sources if specified
    if (sources && sources.length > 0) {
      results = results.filter(l => sources.includes(l.src));
    }

    // Apply limit
    results = results.slice(0, limit);

    return { labels: results };
  }

  // Get count of stored labels
  getLabelCount(): number {
    return this.allLabels.length;
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

  // The Main Event: Broadcast AND Store
  async emitLabel(uri: string, val: string, cid?: string) {
    if (!this.keypair) throw new Error("Server not started: No Keypair");

    const cts = new Date().toISOString();

    const label = {
      ver: 1,
      src: this.did,
      uri: uri,
      val: val,
      neg: false,
      cts: cts,
      ...(cid && { cid }),
    };

    // Sign and Encode
    const toSign = cbor.encode(label);
    const sig = await this.keypair.sign(toSign);

    // Store the label with $bytes-encoded signature for queryLabels (ATProto format)
    const storedLabel: StoredLabel = {
      ...label,
      sig: { $bytes: Buffer.from(sig).toString('base64') },
    };

    // Add to storage
    if (!this.labelsByUri.has(uri)) {
      this.labelsByUri.set(uri, []);
    }
    this.labelsByUri.get(uri)!.push(storedLabel);
    this.allLabels.push(storedLabel);

    // Limit memory usage - keep last 10000 labels
    if (this.allLabels.length > 10000) {
      const removed = this.allLabels.shift();
      if (removed) {
        const uriLabels = this.labelsByUri.get(removed.uri);
        if (uriLabels) {
          const idx = uriLabels.findIndex(l => l.cts === removed.cts && l.val === removed.val);
          if (idx >= 0) uriLabels.splice(idx, 1);
          if (uriLabels.length === 0) this.labelsByUri.delete(removed.uri);
        }
      }
    }

    // Broadcast via WebSocket
    const signedLabel = { ...label, sig };
    const header = { t: "#labels", op: 1 };
    const message = { seq: Date.now(), labels: [signedLabel] };

    const frame = Buffer.concat([
      Buffer.from(cbor.encode(header)),
      Buffer.from(cbor.encode(message))
    ]);

    let sent = 0;
    for (const client of this.wss.clients) {
      if (client.readyState === WebSocket.OPEN) {
        client.send(frame);
        sent++;
      }
    }

    console.log(`🏷️  Label stored & broadcast: ${val} on ${uri.slice(0, 50)}... (${this.allLabels.length} total)`);
  }
}