import { Secp256k1Keypair } from "@atproto/crypto";
import { LabelerServer } from "../labeler-server.js";
import * as dotenv from "dotenv";

dotenv.config();

const run = async () => {
  console.log("\n🔎 DIAGNOSTIC MODE STARTED\n");

  // 1. CHECK IDENTITY
  const hex = process.env.LABELER_PRIVATE_KEY || "";
  const cleanHex = hex.startsWith("0x") ? hex.slice(2) : hex;
  
  if (!cleanHex) {
    console.error("❌ ERROR: LABELER_PRIVATE_KEY is missing in .env");
    process.exit(1);
  }

  try {
    const keypair = await Secp256k1Keypair.import(Buffer.from(cleanHex, "hex"));
    const publicKey = keypair.did();
    
    console.log("✅ Private Key is valid.");
    console.log("🔑 Server Identity (Public Key):");
    console.log(`   ${publicKey}`);
    console.log("\n👉 ACTION: Go to https://plc.directory/ and search for your DID.");
    console.log(`   Verify that 'verificationMethods' contains this EXACT public key: ${publicKey}\n`);
  } catch (e) {
    console.error("❌ ERROR: Private Key is invalid or corrupt.");
    console.error(e);
    process.exit(1);
  }

  // 2. FORCE BROADCAST
  const targetUri = process.argv[2];

  if (!targetUri) {
    console.log("⚠️  No URI provided. Skipping broadcast test.");
    console.log("   Usage: npx tsx src/diagnose.ts <YOUR_POST_URI>");
    return;
  }

  console.log(`📡 Attempting to label: ${targetUri}`);
  
  try {
    const labeler = new LabelerServer(process.env.LABELER_DID || "");
    await labeler.loadKey(process.env.LABELER_PRIVATE_KEY || "");
    
    // Force emit "election" label
    await labeler.emitLabel(targetUri, "election");
    console.log("✅ Broadcast command sent successfully.");
  } catch (e) {
    console.error("❌ Broadcast failed:", e);
  }
};

run();