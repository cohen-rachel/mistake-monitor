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
  mistake_count: number;
  primary_mistake_type_label?: string;
  mistake_type_labels: string[];
  primary_focus_label?: string;
  focus_labels: string[];
}

export interface PracticeSelection {
  topic_key: string;
  topic_text: string;
  is_free_talk: boolean;
  estimated_level?: string;
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

export interface ProgressPoint {
  session_id: number;
  date: string;
  word_count: number;
  mistake_count: number;
  error_rate_per_100_words: number;
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

export interface SpeakingWinItem {
  event_id: number;
  memory_id: number;
  created_at: string;
  summary: string;
  focus_label: string;
  previous_bad_sentence?: string;
  improved_sentence?: string;
  previous_wrong_span?: string;
  suggested_correction?: string;
  reason?: string;
  confidence?: number;
}

export interface InsightsResponse {
  top_mistakes: MistakeCountItem[];
  common_patterns: MistakeCountItem[];
  trends: TrendPoint[];
  recent_mistakes: RecentMistakeItem[];
  progress: ProgressPoint[];
  improvement_banners: string[];
  latest_speaking_win?: SpeakingWinItem | null;
  speaking_win_history: SpeakingWinItem[];
}

export interface PracticePrompt {
  prompt: string;
  expected_answer: string;
}

export interface PracticeResponse {
  mistake_type_code: string;
  prompts: PracticePrompt[];
}

export interface TopicItem {
  key: string;
  title: string;
  prompt: string;
  level: string;
}

export interface TopicListResponse {
  estimated_level: string;
  topics: TopicItem[];
}

export interface TopicAttemptItem {
  id: number;
  session_id: number;
  date: string;
  language: string;
  topic_key: string;
  topic_text: string;
  is_free_talk: boolean;
  estimated_level?: string;
  mistake_count: number;
  transcript: string;
}

export interface TopicHistoryResponse {
  topic_key: string;
  attempts: TopicAttemptItem[];
}

export interface RewriteExerciseResponse {
  source_mistake_id: number;
  language: string;
  mistake_type_code: string;
  mistake_type_label: string;
  original_sentence: string;
  wrong_span?: string;
  expected_correction?: string;
  explanation_short?: string;
}

export interface RewriteSubmitResponse {
  is_correct: boolean;
  score: number;
  feedback: string;
  expected_correction?: string;
}

export interface RewriteStatsItem {
  wrong_span?: string;
  expected_correction?: string;
  attempts: number;
  correct_attempts: number;
  accuracy: number;
  latest_result?: boolean;
  latest_date?: string;
}

export interface RewriteStatsResponse {
  total_attempts: number;
  total_correct: number;
  overall_accuracy: number;
  recent_attempts: RewriteStatsItem[];
}

/* WebSocket transcript chunk message */
export interface TranscriptChunk {
  type: "transcript" | "error" | "stopped";
  text?: string;
  full_text?: string;
  chunk_index: number;
  is_final?: boolean;
  is_provisional?: boolean;
  confidence?: number;
  message?: string;
}

export interface FinalTranscriptionResponse {
  analysis_text: string;
  display_text: string;
  is_provisional: false;
  transcript_source: string;
  average_confidence: number;
  segments: Array<{
    text: string;
    start: number;
    end: number;
    confidence: number;
  }>;
}

export interface TranscriptionConfigResponse {
  live_stt_provider: string;
  final_stt_provider: string;
  skip_final_pass: boolean;
}

export interface UserLanguageProfileOut {
  id: number;
  user_id: number;
  language_code: string;
  display_name: string;
  created_at: string;
}
