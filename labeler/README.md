# Labeler Structure

```
├── src/
│   ├── index.ts          # Entry point. Connects to Jetstream & starts Server.
│   ├── jetstream.ts      # Code that handles the incoming firehose data.
│   ├── labeler-server.ts # Code that runs the WebSocket for Bluesky to listen to.
│   └── db.ts             # Database connection logic.
├── logic/
│   ├── analyzer.py       # Your logic. Reads text, returns label.
│   └── requirements.txt  # Python libraries
├── Dockerfile            # The instruction manual for Fly.io to build your "box".
├── fly.toml              # Fly.io configuration (ports, region).
└── package.json
```


## Summary of Data Flow

1. Bluesky Network emits a post.
2. `jetstream.ts` catches it (resume from `db.ts` cursor).
3. `jetstream.ts` sends text to Python.
4. Python returns a label to `jetstream.ts`.
5. `jetstream.ts` tells `labeler-server.ts` to broadcast.
6. `labeler-server.ts` signs the label with Keypair.
7. `labeler-server.ts` encodes to CBOR and pushes to WebSocket.
8. Bluesky PDS/Relay (connected via `index.ts`) receives the label.

# BELOW IS AN AI SUMMARY

# 🕷️ Bluesky Labeler: Composite Architecture

This repository contains a high-throughput, cross-language labeling service for the Bluesky AT Protocol Firehose. It utilizes a **Composite Architecture** where a fast, asynchronous Node.js server handles network I/O (Jetstream/WebSockets) and passes data to a dedicated Python backend for heavy analysis/logic.

---

## 1. Project Architecture (TypeScript & Python)

Your system is designed to consume the live firehose, process text in a separate Python process (using a high-speed Unix Domain Socket for inter-process communication), and broadcast signed labels back to the network.

### A. Source Code (`/labeler/src`)

| File | Role | Function |
| :--- | :--- | :--- |
| **`index.ts`** | **The Coordinator** | Application entry point. Sets up the HTTP server (for Fly.io health checks), initializes the `LabelerServer` (loading cryptographic keys), and starts the `Jetstream` data stream. |
| **`labeler-server.ts`** | **The Broadcaster/Crypto** | Implements the **WebSocket Server** (`/subscribeLabels`). Handles client connections, signs all outgoing labels with your private key, applies clock skew corrections (`cts`), and encodes data into **DAG-CBOR**. |
| **`jetstream.ts`** | **The Listener/Ingestion** | Connects to the **Bluesky Firehose**. It filters posts, batches text data, sends analysis requests to Python via **Unix Socket**, and calls `labeler.emitLabel()` if a match is found. |
| **`db.ts`** | **Memory/Storage** | Manages the **SQLite database**. It stores the `cursor` (bookmark for the firehose) and the `history` (log of successfully applied labels). |

### B. Utility Scripts (`/labeler/src/scripts`)

| File | Role | Purpose |
| :--- | :--- | :--- |
| **`publish-labels.ts`** | **Network Declaration** | **(Run Once Locally)** Logs into your account and publishes the `app.bsky.labeler.service` record. This tells the network what "election" or "vaccine" means (severity, descriptions, etc.). |
| **`check-history.ts`** | **Database Verification** | Reads the live `labeler.db` history file (accepts `N` as argument) to verify that your labeler found matches and stored them correctly. |
| **`get-public-key.ts`**| **Key Verification** | Derives and prints the public key (`did:key:zQ3sh...`) corresponding to the private key in your environment variables. Essential for checking against the PLC Directory. |

---

## 2. AT Protocol Fundamentals & Glossary

| Term | Explanation | Relevance to Project |
| :--- | :--- | :--- |
| **DID** (Decentralized Identifier) | Your account's unique, portable identity (`did:plc:`, `did:web:`). | The network validates your signature against the Public Key listed in your DID Document. |
| **URI** (AT Protocol URI) | The unique, permanent address of a specific record (post, follow, etc.). | Used as the `uri` field in the label object; must end in the **Record Key (rkey)**. |
| **CID** (Content Identifier) | A cryptographic hash of a record's content *at a specific point in time*. | Used as the optional `cid` field in the label object to pin the label to a specific version of the post (recommended for moderation). |

---

## 3. Deployment & Monitoring (Fly.io)

### A. Deployment Structure

The deployment uses the following custom files to run two services in one container:

* **`Dockerfile`**: Builds the environment (Installs Node, Python, and dependencies).
* **`start.sh`**: The core execution script. It starts the Python server (FastAPI) on a **Unix Domain Socket** (`/tmp/analyzer.sock`) and then launches the Node server via `exec npm start`.
* **`fly.toml`**: Configures the port, volume mount (`/data`), and health checks.

### B. Verification and Debugging Cheat Sheet

Since the machine is live and inaccessible via direct terminal input, use the `fly ssh console -C` command to remotely execute diagnostic scripts.

| Task | Command | Purpose |
| :--- | :--- | :--- |
| **Run History Check** | `fly ssh console -C "npx tsx src/scripts/check-history.ts 15" -a <APP_NAME>` | Runs the history checker on the live server and prints the last 15 labeled posts. |
| **Check Keys & Broadcast** | `fly ssh console -C "npx tsx src/scripts/diagnose.ts <YOUR_POST_URI>" -a <APP_NAME>` | **The Final Test.** Checks that the server key is valid and forces a live label broadcast for a specific post. |
| **Stream Live Logs** | `fly logs` | Essential for real-time debugging. Look for `🔌 Client connected...` and `[MATCH] Labeled...`. |
| **Check Public Status** | `fly status` | Confirms the machine is in the `started` state and its region. |
| **Check API Acceptance** | `https://public.api.bsky.app/xrpc/app.bsky.feed.getPostThread?uri=YOUR_URI_HERE` | Bypasses the app cache to confirm if the **network** accepted your label (`labels: [...]` will be populated). |