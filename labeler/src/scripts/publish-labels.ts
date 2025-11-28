#!/usr/bin/env node
/**
 * Dynamic Label Publisher for Bluesky
 * 
 * This script reads the generated labels from the Python analyzer
 * and publishes them to the Bluesky network.
 * 
 * Usage:
 *   npx tsx src/scripts/publish-labels.ts
 * 
 * Prerequisites:
 *   1. Run the Python analyzer at least once to generate labels
 *   2. Set BSKY_HANDLE and BSKY_PASSWORD in .env
 */

import { AtpAgent } from "@atproto/api";
import * as dotenv from "dotenv";

dotenv.config();

interface LabelLocale {
  lang: string;
  name: string;
  description: string;
}

interface LabelDefinition {
  identifier: string;
  severity: string;
  blurs: string;
  defaultSetting: string;
  locales: LabelLocale[];
}

interface GeneratedLabels {
  labelValues: string[];
  labelValueDefinitions: LabelDefinition[];
  clusters?: any[];
  generated_at?: string;
}

// Static label definitions - only 2 labels: misinformation and ai-generated
function getStaticLabels(): GeneratedLabels {
  return {
    labelValues: ["misinformation", "ai-generated"],
    labelValueDefinitions: [
      {
        identifier: "misinformation",
        severity: "alert",
        blurs: "content",
        defaultSetting: "warn",
        locales: [
          {
            lang: "en",
            name: "Misinformation",
            description: "This post contains potential misinformation based on Community Notes fact-checks"
          }
        ]
      },
      {
        identifier: "ai-generated",
        severity: "alert",
        blurs: "content",
        defaultSetting: "warn",
        locales: [
          {
            lang: "en",
            name: "AI Generated",
            description: "This post contains AI-generated content that may be misleading"
          }
        ]
      }
    ],
    generated_at: new Date().toISOString()
  };
}

async function run() {
  console.log("\n🏷️  LABEL PUBLISHER\n");

  // Validate credentials
  if (!process.env.BSKY_HANDLE || !process.env.BSKY_PASSWORD) {
    console.error("❌ ERROR: BSKY_HANDLE and BSKY_PASSWORD are required in .env");
    process.exit(1);
  }

  // Use static 2-label configuration
  const labels = getStaticLabels();

  // Display labels to be published
  console.log("📋 Labels to publish:");
  for (const def of labels.labelValueDefinitions) {
    console.log(`   - ${def.identifier}: ${def.locales[0]?.name || 'No name'}`);
  }
  console.log();

  // Authenticate with Bluesky
  console.log("🔐 Authenticating with Bluesky...");
  const agent = new AtpAgent({ service: "https://bsky.social" });

  try {
    await agent.login({
      identifier: process.env.BSKY_HANDLE!,
      password: process.env.BSKY_PASSWORD!,
    });
    console.log(`✅ Authenticated as ${process.env.BSKY_HANDLE}`);
  } catch (error) {
    console.error("❌ Authentication failed:", error);
    process.exit(1);
  }

  // Publish labeler service record
  console.log("\n📤 Publishing labeler service record...");

  try {
    await agent.com.atproto.repo.putRecord({
      repo: agent.session?.did ?? "",
      collection: "app.bsky.labeler.service",
      rkey: "self",
      record: {
        createdAt: new Date().toISOString(),
        policies: {
          labelValues: labels.labelValues,
          labelValueDefinitions: labels.labelValueDefinitions
        }
      },
    });

    console.log("✅ Labels published successfully!\n");
    console.log("📊 Summary:");
    console.log(`   - Total labels: ${labels.labelValues.length}`);
    console.log(`   - DID: ${agent.session?.did}`);
    console.log("\n🎉 Your labeler is now configured with 2 labels: misinformation & ai-generated\n");

  } catch (error) {
    console.error("❌ Failed to publish labels:", error);
    process.exit(1);
  }
}

run().catch(console.error);
