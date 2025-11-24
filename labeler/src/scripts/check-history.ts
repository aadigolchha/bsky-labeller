import * as db from "../db.js";

// Get the last 20 labels
const labels = db.getRecentLabels(20);

console.table(labels);