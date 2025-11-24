import { Secp256k1Keypair } from "@atproto/crypto";
import * as dotenv from "dotenv";
dotenv.config();

const run = async () => {
  const hex = process.env.LABELER_PRIVATE_KEY || "";
  const cleanHex = hex.startsWith("0x") ? hex.slice(2) : hex;
  const keypair = await Secp256k1Keypair.import(Buffer.from(cleanHex, "hex"));
  console.log("Your DID Public Key:", keypair.did());
};
run();