/* Shared TypeScript types matching backend Pydantic schemas. */

export interface MistakeType {
  id: number;
  code: string;
  label: string;
  description?: string;
}

export interface MistakeOut {
  id: number;
  session_id: number;
  mistake_type: MistakeType;
  transcript_span?: string;
  start_char?: number;
  end_char?: number;
  suggested_correction?: string;
  explanation_short?: string;
  confidence?: number;
  canonical_example?: string;
  stt_uncertain: boolean;
  uncertain: boolean;
  uncertain_reason?: string;
}

export interface TranscriptOut {
  id: number;
  session_id: number;
  raw_text: string;
  tokens_with_timestamps?: Array<{
    text: string;
    start: number;
    end: number;
    confidence: number;
  }>;
}

export interface SessionOut {
  id: number;
  user_id?: number;
  created_at: string;
  language: string;
  audio_meta?: Record<string, unknown>;
  stt_provider?: string;
  stt_confidence_summary?: number;
  status: string;
}

export interface SessionDetailOut extends SessionOut {
  transcript?: TranscriptOut;
  mistakes: MistakeOut[];
}

export interface SessionListOut {
  sessions: SessionOut[];
}

export interface AnalyzeRequest {
  session_id: number;
  transcript_text?: string;
}

export interface AnalyzeResponse {
  session_id: number;
  mistakes: MistakeOut[];
}

export interface MistakeCountItem {
  code: string;
  label: string;
  count: number;
  description?: string;
  recent_mistake_summary?: string;
}

export interface TrendPoint {
  session_id: number;
  date: string;
  mistake_type_code: string;
  count: number;
}

export interface RecentMistakeItem {
  id: number;
  session_id: number;
  date: string;
  mistake_type_code: string;
  mistake_type_label: string;
  transcript_span?: string;
  suggested_correction?: string;
  explanation_short?: string;
  original_sentence?: string;
  corrected_sentence?: string;
}

export interface InsightsResponse {
  top_mistakes: MistakeCountItem[];
  trends: TrendPoint[];
  recent_mistakes: RecentMistakeItem[];
}

export interface PracticePrompt {
  prompt: string;
  expected_answer: string;
}

export interface PracticeResponse {
  mistake_type_code: string;
  prompts: PracticePrompt[];
}

/* WebSocket transcript chunk message */
export interface TranscriptChunk {
  type: "transcript" | "error" | "stopped";
  text?: string;
  full_text?: string;
  chunk_index: number;
  is_final?: boolean;
  confidence?: number;
  message?: string;
}
