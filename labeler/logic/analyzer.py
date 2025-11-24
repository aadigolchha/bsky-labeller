# This is a micro-webserver. It listens for text and returns a label.

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class PostPayload(BaseModel):
    text: str
    did: str

@app.post("/analyze")
async def analyze(payload: PostPayload):
    # YOUR LOGIC GOES HERE
    text = payload.text.lower()
    
    #if "election news" in text:
    #    return {"label": "election-label", "score": 1.0, "text": text}

    if "election news" in text:
        return {"label": "election", "score": 1.0, "text": text}

    if "vaccination" in text:
        return {"label": "vaccine", "score": 1.0, "text": text}
    
    return {"label": None}

if __name__ == "__main__":
    # Run on port 5001 or any free port
    uvicorn.run(app, host="0.0.0.0", port=5001)