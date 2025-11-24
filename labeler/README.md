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

