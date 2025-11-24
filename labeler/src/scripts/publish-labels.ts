import { AtpAgent } from "@atproto/api";
import * as dotenv from "dotenv";

dotenv.config();

const run = async () => {
  const agent = new AtpAgent({ service: "https://bsky.social" });

  await agent.login({
    identifier: process.env.BSKY_HANDLE!,
    password: process.env.BSKY_PASSWORD!,
  });

  console.log("Publishing Labeler Service Record...");

  await agent.com.atproto.repo.putRecord({
    repo: agent.session?.did ?? "",
    collection: "app.bsky.labeler.service",
    rkey: "self",
    record: {
      createdAt: new Date().toISOString(),
      policies: {
        // PART 1: The "Menu List" (Strings Only)
        // This was the cause of your error. It must be just strings.
        labelValues: ["election", "vaccine"],

        // PART 2: The "Detailed Definitions" (Configuration)
        labelValueDefinitions: [
          // DEFINITION 1: ELECTION
          {
            identifier: "election",
            severity: "inform",
            blurs: "none",
            defaultSetting: "warn",
            locales: [
              { lang: "en", name: "Election News", description: "Posts related to election news." }
            ]
          },
          // DEFINITION 2: VACCINE
          {
            identifier: "vaccine",
            severity: "inform",
            blurs: "none",
            defaultSetting: "warn",
            locales: [
              { lang: "en", name: "Vaccine Info", description: "Discussion regarding vaccines." }
            ]
          }
        ]
      }
    },
  });

  console.log("✅ Labels published successfully!");
};

run();