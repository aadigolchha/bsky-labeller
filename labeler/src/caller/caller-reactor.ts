/**
 * Caller Reactor
 *
 * Takes the report payload + chosen label and executes the action:
 * creates/attaches the label via the existing LabelerServer.
 * Logs everything for debugging (visible on fly.io and in terminal).
 */

import type { LabelerServer } from "@skyware/labeler";
import type { ReportPayload } from "./caller-analyzer.js";

export type ReactorResult = {
  success: boolean;
  label?: string;
  error?: string;
};

/**
 * Executes the label action: creates the label on the given post via the labeler.
 * Logs request, label, and outcome for debugging.
 */
export async function executeLabelAction(
  payload: ReportPayload,
  label: string,
  labeler: LabelerServer
): Promise<ReactorResult> {
  const logPrefix = "[CALLER-REACTOR]";

  console.log(`${logPrefix} Executing label action`);
  console.log(`${logPrefix}   uri: ${payload.uri}`);
  console.log(`${logPrefix}   cid: ${payload.cid ?? "(none)"}`);
  console.log(`${logPrefix}   reason: ${payload.reason ?? "(none)"}`);
  console.log(`${logPrefix}   label: ${label}`);

  try {
    await labeler.createLabel({
      uri: payload.uri,
      cid: payload.cid,
      val: label,
    });

    console.log(`${logPrefix} ✅ Label applied successfully`);
    console.log(`${logPrefix}   Post ${payload.uri} labeled as "${label}"`);

    return { success: true, label };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error(`${logPrefix} ❌ Failed to apply label`);
    console.error(`${logPrefix}   uri: ${payload.uri}`);
    console.error(`${logPrefix}   label: ${label}`);
    console.error(`${logPrefix}   error: ${message}`);
    if (err instanceof Error && err.stack) {
      console.error(`${logPrefix}   stack: ${err.stack}`);
    }

    return { success: false, error: message };
  }
}
