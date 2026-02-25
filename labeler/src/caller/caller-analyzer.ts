/**
 * Caller Analyzer
 *
 * Parses the incoming report payload and decides which label to apply.
 * For now: randomly picks "misinformation" or "ai-generated".
 * Later: can be replaced with ML / rule-based logic.
 */

export type ReportPayload = {
  uri: string;
  cid?: string;
  reason?: string;
};

export type AnalyzerResult = {
  label: "misinformation" | "ai-generated";
};

const LABELS: AnalyzerResult["label"][] = ["misinformation", "ai-generated"];

/**
 * Determines the label to apply for the given report.
 * Currently returns a random choice; replace with real logic later.
 */
export async function analyzeReport(payload: ReportPayload): Promise<AnalyzerResult> {
  const label = LABELS[Math.floor(Math.random() * LABELS.length)] as AnalyzerResult["label"];
  return { label };
}
