import * as db from "../db.js";

// 1. Read the command line argument (default to 10 if missing)
// process.argv[0] is 'node', [1] is the script path, [2] is your argument
const args = process.argv.slice(2);
const limit = args.length > 0 ? parseInt(args[0], 10) : 10;

if (isNaN(limit)) {
  console.error("❌ Error: Argument must be a number");
  process.exit(1);
}

console.log(`\n🔍 Fetching the last ${limit} labels from history...\n`);

// 2. Fetch from DB
const labels = db.getRecentLabels(limit);

// 3. Display
if (labels.length === 0) {
  console.log("No labels found yet. Waiting for matches...");
} else {
  console.table(labels);
}