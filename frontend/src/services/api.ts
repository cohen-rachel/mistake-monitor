/**
 * API service layer — all backend calls in one place.
 * Separated from UI for easy migration to React Native later.
 */

import type {
  SessionDetailOut,
  SessionListOut,
  AnalyzeResponse,
  FinalTranscriptionResponse,
  InsightsResponse,
  TopicListResponse,
  TopicHistoryResponse,
  PracticeSelection,
  RewriteExerciseResponse,
  RewriteStatsResponse,
  RewriteSubmitResponse,
  TranscriptionConfigResponse,
} from "../types";
import type { UserLanguageProfileOut } from "../types";

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
    let message = `API error ${resp.status}`;
    try {
      const parsed = JSON.parse(text);
      if (Array.isArray(parsed?.detail) && parsed.detail[0]?.msg) {
        message = parsed.detail[0].msg;
      } else if (typeof parsed?.detail === "string") {
        message = parsed.detail;
      } else if (text) {
        message = `${message}: ${text}`;
      }
    } catch {
      if (text) {
        message = `${message}: ${text}`;
      }
    }
    const error = new Error(message) as Error & { status?: number };
    error.status = resp.status;
    throw error;
  }
  return resp.json();
}

// ---------- Sessions ----------

export async function createSessionWithTranscript(
  transcriptText: string,
  languageProfileId: number,
  practice?: PracticeSelection
): Promise<SessionDetailOut> {
  const form = new FormData();
  form.append("transcript_text", transcriptText);
  form.append("language_profile_id", String(languageProfileId));
  if (practice) {
    form.append("practice_topic_key", practice.topic_key);
    form.append("practice_topic_text", practice.topic_text);
    form.append("is_free_talk", String(practice.is_free_talk));
    if (practice.estimated_level) {
      form.append("estimated_level", practice.estimated_level);
    }
  }
  return request<SessionDetailOut>("/sessions", { method: "POST", body: form });
}

export async function createSessionWithAudio(
  audioFile: File,
  languageProfileId: number,
  practice?: PracticeSelection
): Promise<SessionDetailOut> {
  const form = new FormData();
  form.append("audio_file", audioFile);
  form.append("language_profile_id", String(languageProfileId));
  if (practice) {
    form.append("practice_topic_key", practice.topic_key);
    form.append("practice_topic_text", practice.topic_text);
    form.append("is_free_talk", String(practice.is_free_talk));
    if (practice.estimated_level) {
      form.append("estimated_level", practice.estimated_level);
    }
  }
  return request<SessionDetailOut>("/sessions", { method: "POST", body: form });
}

export async function finalizeRecordedAudio(
  audioFile: File,
  language?: string
): Promise<FinalTranscriptionResponse> {
  const form = new FormData();
  form.append("audio_file", audioFile);
  if (language) {
    form.append("language", language);
  }
  return request<FinalTranscriptionResponse>("/transcribe/finalize", {
    method: "POST",
    body: form,
  });
}

export async function getTranscriptionConfig(): Promise<TranscriptionConfigResponse> {
  return request<TranscriptionConfigResponse>("/transcribe/config");
}

export async function listSessions(
  languageProfileId?: number
): Promise<SessionListOut> {
  const query = languageProfileId
    ? `?language_profile_id=${encodeURIComponent(String(languageProfileId))}`
    : "";
  return request<SessionListOut>(`/sessions${query}`);
}

export async function getSession(id: number): Promise<SessionDetailOut> {
  return request<SessionDetailOut>(`/sessions/${id}`);
}

// ---------- Analyze ----------

export async function analyzeSession(
  sessionId: number,
  transcriptText?: string,
  allowLanguageMismatch: boolean = false
): Promise<AnalyzeResponse> {
  return request<AnalyzeResponse>("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      transcript_text: transcriptText,
      allow_language_mismatch: allowLanguageMismatch,
    }),
  });
}

// ---------- Insights ----------

export async function getInsights(
  topK: number = 10,
  recentN: number = 20,
  languageProfileId: number 
): Promise<InsightsResponse> {
  return request<InsightsResponse>(
    `/insights?top_k=${topK}&recent_n=${recentN}&language_profile_id=${encodeURIComponent(String(languageProfileId))}`
  );
}

// ---------- Topic Practice ----------

export async function getTopics(
  language_code: string,
  userId: number = 1
): Promise<TopicListResponse> {
  return request<TopicListResponse>(
    `/topics?language_code=${encodeURIComponent(language_code)}&user_id=${userId}`
  );
}

export async function getTopicHistory(
  topicKey: string,
  language_code: string,
  userId: number = 1
): Promise<TopicHistoryResponse> {
  return request<TopicHistoryResponse>(
    `/topics/history?topic_key=${encodeURIComponent(topicKey)}&language_code=${encodeURIComponent(language_code)}&user_id=${userId}`
  );
}

// ---------- Rewrite Training ----------

export async function getRewriteExercise(
  language_code: string,
  userId: number = 1,
  excludeMistakeIds?: number[]
): Promise<RewriteExerciseResponse> {
  const params = new URLSearchParams({
    language_code,
    user_id: String(userId),
  });
  excludeMistakeIds?.forEach((id) => params.append("exclude_mistake_ids", String(id)));
  return request<RewriteExerciseResponse>(`/rewrite/next?${params.toString()}`);
}

export async function submitRewriteExercise(
  payload: {
    user_id: number;
    language_code: string;
    source_mistake_id: number;
    original_sentence: string;
    wrong_span?: string;
    expected_correction?: string;
    user_rewrite: string;
  }
): Promise<RewriteSubmitResponse> {
  return request<RewriteSubmitResponse>("/rewrite/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function getRewriteStats(
  language_code: string,
  userId: number = 1
): Promise<RewriteStatsResponse> {
  return request<RewriteStatsResponse>(
    `/rewrite/stats?language_code=${encodeURIComponent(language_code)}&user_id=${userId}`
  );
}

// ---------- WebSocket ----------

export function createTranscribeSocket(language?: string): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host;
  const params = new URLSearchParams();
  if (language) {
    params.set("language", language);
  }
  const query = params.toString();
  return new WebSocket(`${protocol}//${host}/api/transcribe/stream${query ? `?${query}` : ""}`);
}

export async function getUserLanguageProfiles(): Promise<UserLanguageProfileOut[]> {
  return request<UserLanguageProfileOut[]>("/user/language_profiles");
}

export async function getCurrentLanguageProfile(): Promise<UserLanguageProfileOut | null> {
  try {
    return await request<UserLanguageProfileOut>("/user/language_profiles/current");
  } catch {
    return null;
  }
}

export async function setCurrentLanguageProfile(
  profileId: number
): Promise<UserLanguageProfileOut> {
  return request<UserLanguageProfileOut>("/user/language_profiles/set_current", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profile_id: profileId }),
  });
}

export const api = {
  getUserLanguageProfiles,
  getCurrentLanguageProfile,
  setCurrentLanguageProfile,
};
