/**
 * API service layer — all backend calls in one place.
 * Separated from UI for easy migration to React Native later.
 */

import type {
  SessionDetailOut,
  SessionListOut,
  AnalyzeResponse,
  InsightsResponse,
  PracticeResponse,
} from "../types";

const API_BASE = "/api";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...(options?.headers || {}),
    },
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`API error ${resp.status}: ${text}`);
  }
  return resp.json();
}

// ---------- Sessions ----------

export async function createSessionWithTranscript(
  transcriptText: string,
  language: string = "en"
): Promise<SessionDetailOut> {
  const form = new FormData();
  form.append("transcript_text", transcriptText);
  form.append("language", language);
  return request<SessionDetailOut>("/sessions", { method: "POST", body: form });
}

export async function createSessionWithAudio(
  audioFile: File,
  language: string = "en"
): Promise<SessionDetailOut> {
  const form = new FormData();
  form.append("audio_file", audioFile);
  form.append("language", language);
  return request<SessionDetailOut>("/sessions", { method: "POST", body: form });
}

export async function listSessions(): Promise<SessionListOut> {
  return request<SessionListOut>("/sessions");
}

export async function getSession(id: number): Promise<SessionDetailOut> {
  return request<SessionDetailOut>(`/sessions/${id}`);
}

// ---------- Analyze ----------

export async function analyzeSession(
  sessionId: number,
  transcriptText?: string
): Promise<AnalyzeResponse> {
  return request<AnalyzeResponse>("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      transcript_text: transcriptText,
    }),
  });
}

// ---------- Insights ----------

export async function getInsights(
  topK: number = 10,
  recentN: number = 20
): Promise<InsightsResponse> {
  return request<InsightsResponse>(
    `/insights?top_k=${topK}&recent_n=${recentN}`
  );
}

// ---------- Practice ----------

export async function getPractice(
  mistakeTypeCode: string,
  count: number = 3
): Promise<PracticeResponse> {
  return request<PracticeResponse>("/practice", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      mistake_type_code: mistakeTypeCode,
      count,
    }),
  });
}

// ---------- WebSocket ----------

export function createTranscribeSocket(language: string = "en"): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host;
  const encodedLanguage = encodeURIComponent(language);
  return new WebSocket(
    `${protocol}//${host}/api/transcribe/stream?language=${encodedLanguage}`
  );
}
