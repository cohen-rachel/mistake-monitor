import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import AudioRecorder from "../components/AudioRecorder";
import TranscriptDisplay from "../components/TranscriptDisplay";
import MistakeCard from "../components/MistakeCard";
import {
  createSessionWithTranscript,
  createSessionWithAudio,
  analyzeSession,
  getTopics,
  getTopicHistory,
} from "../services/api";
import type {
  MistakeOut,
  TopicItem,
  TopicAttemptItem,
  PracticeSelection,
} from "../types";
import { useLanguageContext } from "../contexts/LanguageContext";

const sectionStyle: React.CSSProperties = {
  background: "#fff",
  borderRadius: 12,
  padding: 24,
  border: "1px solid #e2e8f0",
  marginBottom: 24,
};

const tabBarStyle: React.CSSProperties = {
  display: "flex",
  gap: 0,
  marginBottom: 24,
};

const tabStyle = (active: boolean): React.CSSProperties => ({
  padding: "10px 24px",
  border: "1px solid #e2e8f0",
  background: active ? "#4338ca" : "#fff",
  color: active ? "#fff" : "#475569",
  cursor: "pointer",
  fontWeight: 600,
  fontSize: 14,
  borderRadius: 0,
});

const btnPrimary: React.CSSProperties = {
  padding: "10px 24px",
  borderRadius: 8,
  border: "none",
  background: "#059669",
  color: "#fff",
  fontWeight: 600,
  fontSize: 15,
  cursor: "pointer",
};

const btnDisabled: React.CSSProperties = {
  ...btnPrimary,
  background: "#94a3b8",
  cursor: "not-allowed",
};

function formatDuration(totalSeconds: number): string {
  const m = Math.floor(totalSeconds / 60)
    .toString()
    .padStart(2, "0");
  const s = (totalSeconds % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

export default function Landing() {
  const { currentLanguageProfile, isLoadingLanguage } = useLanguageContext();
  const [tab, setTab] = useState<"record" | "upload">("record");

  // Topic practice state
  const [topics, setTopics] = useState<TopicItem[]>([]);
  const [estimatedLevel, setEstimatedLevel] = useState("beginner");
  const [selectedTopicKey, setSelectedTopicKey] = useState("free_talk");
  const [topicHistory, setTopicHistory] = useState<TopicAttemptItem[]>([]);
  const [topicsLoading, setTopicsLoading] = useState(false);

  // Record state
  const [isRecording, setIsRecording] = useState(false);
  const [elapsedSec, setElapsedSec] = useState(0);
  const [liveTranscript, setLiveTranscript] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [mistakes, setMistakes] = useState<MistakeOut[]>([]);
  const [statusMsg, setStatusMsg] = useState("");

  // Upload state
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadAnalyzing, setUploadAnalyzing] = useState(false);
  const [uploadMistakes, setUploadMistakes] = useState<MistakeOut[]>([]);
  const [uploadTranscript, setUploadTranscript] = useState("");
  const [uploadStatus, setUploadStatus] = useState("");

  const fileInputRef = useRef<HTMLInputElement>(null);

  const selectedTopic = useMemo(
    () => topics.find((t) => t.key === selectedTopicKey) ?? null,
    [topics, selectedTopicKey]
  );

  const buildPracticeSelection = useCallback((): PracticeSelection => {
    if (selectedTopicKey === "free_talk" || !selectedTopic) {
      return {
        topic_key: "free_talk",
        topic_text: "Speak freely about anything you want with no fixed prompt.",
        is_free_talk: true,
        estimated_level: estimatedLevel,
      };
    }
    return {
      topic_key: selectedTopic.key,
      topic_text: selectedTopic.prompt,
      is_free_talk: false,
      estimated_level: estimatedLevel,
    };
  }, [estimatedLevel, selectedTopic, selectedTopicKey]);

  useEffect(() => {
    if (!currentLanguageProfile) return;

    let cancelled = false;
    const loadTopics = async () => {
      setTopicsLoading(true);
      try {
        const data = await getTopics(currentLanguageProfile.language_code);
        if (cancelled) return;
        setTopics(data.topics);
        setEstimatedLevel(data.estimated_level);
        if (!data.topics.some((t) => t.key === selectedTopicKey)) {
          setSelectedTopicKey(data.topics[0]?.key ?? "free_talk");
        }
      } catch {
        if (!cancelled) {
          setTopics([]);
          setEstimatedLevel("beginner");
          setSelectedTopicKey("free_talk");
        }
      } finally {
        if (!cancelled) setTopicsLoading(false);
      }
    };
    loadTopics();
    return () => {
      cancelled = true;
    };
  }, [currentLanguageProfile, selectedTopicKey]);

  useEffect(() => {
    if (!currentLanguageProfile) return;

    let cancelled = false;
    const loadHistory = async () => {
      if (!selectedTopicKey || selectedTopicKey === "free_talk") {
        setTopicHistory([]);
        return;
      }
      try {
        const data = await getTopicHistory(
          selectedTopicKey,
          currentLanguageProfile.language_code
        );
        if (!cancelled) {
          setTopicHistory(data.attempts);
        }
      } catch {
        if (!cancelled) setTopicHistory([]);
      }
    };
    loadHistory();
    return () => {
      cancelled = true;
    };
  }, [currentLanguageProfile, selectedTopicKey]);

  useEffect(() => {
    if (!isRecording) return;
    const id = window.setInterval(() => {
      setElapsedSec((prev) => prev + 1);
    }, 1000);
    return () => window.clearInterval(id);
  }, [isRecording]);

  const durationHint = useMemo(() => {
    if (!isRecording) return "";
    if (elapsedSec < 30) return "Keep speaking. Target at least 30 seconds.";
    if (elapsedSec <= 60) return "Great length. You are in the 30-60 second target range.";
    return "You can stop anytime. You already exceeded the target range.";
  }, [elapsedSec, isRecording]);

  const handleChunk = useCallback((text: string, fullText?: string) => {
    if (fullText) {
      setLiveTranscript(fullText);
      return;
    }
    setLiveTranscript((prev) => `${prev} ${text}`.trim());
  }, []);

  const handleStatusChange = useCallback((recording: boolean) => {
    setIsRecording(recording);
    if (recording) {
      setElapsedSec(0);
      setStatusMsg("");
      setMistakes([]);
      setLiveTranscript("");
    }
  }, []);

  const handleRandomizeTopic = () => {
    const selectable = topics.filter((t) => t.key !== "free_talk");
    if (selectable.length === 0) return;
    const next = selectable[Math.floor(Math.random() * selectable.length)];
    setSelectedTopicKey(next.key);
  };

  const handleAnalyzeRecording = async () => {
    if (!currentLanguageProfile) {
      setStatusMsg("Select a language profile first.");
      return;
    }
    const fullText = liveTranscript.trim();
    if (!fullText) {
      setStatusMsg("No transcript to analyze. Record something first.");
      return;
    }

    setAnalyzing(true);
    setStatusMsg("Saving session...");
    try {
      const session = await createSessionWithTranscript(
        fullText,
        currentLanguageProfile.id,
        buildPracticeSelection()
      );
      setStatusMsg("Analyzing...");
      const result = await analyzeSession(session.id);
      setMistakes(result.mistakes);
      setStatusMsg(
        result.mistakes.length > 0
          ? `Found ${result.mistakes.length} mistake(s).`
          : "No mistakes found!"
      );
      if (selectedTopicKey !== "free_talk") {
        const updated = await getTopicHistory(
          selectedTopicKey,
          currentLanguageProfile.language_code
        );
        setTopicHistory(updated.attempts);
      }
    } catch (err: any) {
      setStatusMsg(`Error: ${err.message}`);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleReset = () => {
    setLiveTranscript("");
    setMistakes([]);
    setStatusMsg("");
    setElapsedSec(0);
  };

  const handleUploadAnalyze = async () => {
    if (!currentLanguageProfile) {
      setUploadStatus("Select a language profile first.");
      return;
    }
    if (!uploadFile) return;
    setUploadAnalyzing(true);
    setUploadStatus("Uploading and analyzing...");
    try {
      const session = await createSessionWithAudio(
        uploadFile,
        currentLanguageProfile.id,
        buildPracticeSelection()
      );
      setUploadTranscript(session.transcript?.raw_text || "");
      setUploadMistakes(session.mistakes || []);
      setUploadStatus(
        session.mistakes.length > 0
          ? `Found ${session.mistakes.length} mistake(s).`
          : "No mistakes found!"
      );
    } catch (err: any) {
      setUploadStatus(`Error: ${err.message}`);
    } finally {
      setUploadAnalyzing(false);
    }
  };

  return (
    <div>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 16 }}>
        Language Tutor
      </h1>
      <p style={{ color: "#64748b", marginBottom: 24 }}>
        Practice with a suggested topic or choose Free Talk, then get analysis and
        track improvement over time.
      </p>

      {isLoadingLanguage ? (
        <p>Loading language profiles...</p>
      ) : currentLanguageProfile ? (
        <div style={tabBarStyle}>
          <button
            style={{ ...tabStyle(tab === "record"), borderRadius: "8px 0 0 8px" }}
            onClick={() => setTab("record")}
          >
            Real-time Speaking
          </button>
          <button
            style={{ ...tabStyle(tab === "upload"), borderRadius: "0 8px 8px 0" }}
            onClick={() => setTab("upload")}
          >
            Upload Audio File
          </button>
        </div>
      ) : null}

      {isLoadingLanguage ? null : currentLanguageProfile ? (
        <div style={{ ...sectionStyle, marginBottom: 16 }}>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center" }}>
            <div style={{ fontSize: 13, color: "#475569" }}>
              Estimated level: <strong>{estimatedLevel}</strong>
            </div>
          </div>

          <div style={{ marginTop: 12, display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <label style={{ fontSize: 14, color: "#475569" }}>Practice topic:</label>
            <select
              value={selectedTopicKey}
              onChange={(e) => setSelectedTopicKey(e.target.value)}
              style={{
                minWidth: 260,
                padding: "8px 10px",
                borderRadius: 8,
                border: "1px solid #cbd5e1",
                background: "#fff",
                color: "#1e293b",
              }}
              disabled={topicsLoading || topics.length === 0}
            >
              {topics.map((t) => (
                <option key={t.key} value={t.key}>
                  {t.title}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={handleRandomizeTopic}
              style={{ ...btnPrimary, background: "#2563eb", padding: "8px 12px" }}
              disabled={topics.length <= 1}
            >
              Randomize
            </button>
          </div>

          {selectedTopic && (
            <p style={{ marginTop: 10, color: "#334155", fontSize: 14 }}>
              <strong>Prompt:</strong> {selectedTopic.prompt}
            </p>
          )}

          {selectedTopicKey !== "free_talk" && topicHistory.length > 0 && (
            <div style={{ marginTop: 12, padding: 12, borderRadius: 8, background: "#f8fafc", border: "1px solid #e2e8f0" }}>
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 6, color: "#334155" }}>
                Previous attempts for this topic
              </div>
              {topicHistory.slice(0, 3).map((attempt) => (
                <div key={attempt.id} style={{ fontSize: 13, color: "#475569", marginBottom: 4 }}>
                  {attempt.date} · Mistakes: {attempt.mistake_count}
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div style={{ ...sectionStyle, marginBottom: 16 }}>
          <p style={{ color: "#64748b", margin: 0 }}>
            No language profile found yet. A default profile should appear after backend restart.
          </p>
        </div>
      )}

      {tab === "record" && currentLanguageProfile && (
        <div>
          <div style={sectionStyle}>
            <div
              style={{
                display: "flex",
                gap: 12,
                alignItems: "center",
                marginBottom: 12,
                flexWrap: "wrap",
              }}
            >
              <AudioRecorder
                onChunk={handleChunk}
                onStatusChange={handleStatusChange}
                language={currentLanguageProfile.language_code}
              />
              <div style={{ fontSize: 14, color: "#475569" }}>
                Timer: <strong>{formatDuration(elapsedSec)}</strong>
              </div>
              {!isRecording && liveTranscript.length > 0 && (
                <button
                  onClick={handleAnalyzeRecording}
                  disabled={analyzing}
                  style={analyzing ? btnDisabled : btnPrimary}
                >
                  {analyzing ? "Analyzing..." : "Analyze"}
                </button>
              )}
              {!isRecording && liveTranscript.length > 0 && (
                <button
                  onClick={handleReset}
                  style={{
                    ...btnPrimary,
                    background: "#64748b",
                  }}
                >
                  Reset
                </button>
              )}
            </div>

            {durationHint && (
              <p style={{ marginBottom: 8, fontSize: 13, color: "#475569" }}>{durationHint}</p>
            )}

            <TranscriptDisplay transcript={liveTranscript} isRecording={isRecording} />

            {statusMsg && (
              <p style={{ marginTop: 12, fontSize: 14, color: "#475569" }}>
                {statusMsg}
              </p>
            )}
          </div>

          {mistakes.length > 0 && (
            <div>
              <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 12 }}>
                Analysis Results
              </h2>
              {mistakes.map((m) => (
                <MistakeCard key={m.id} mistake={m} />
              ))}
            </div>
          )}
        </div>
      )}

      {tab === "upload" && currentLanguageProfile && (
        <div>
          <div style={sectionStyle}>
            <div
              style={{
                border: "2px dashed #cbd5e1",
                borderRadius: 8,
                padding: 32,
                textAlign: "center",
                marginBottom: 16,
                cursor: "pointer",
                background: uploadFile ? "#f0fdf4" : "#f8fafc",
              }}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                style={{ display: "none" }}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) setUploadFile(file);
                }}
              />
              {uploadFile ? (
                <p style={{ fontWeight: 600, color: "#166534" }}>
                  {uploadFile.name} ({(uploadFile.size / 1024).toFixed(0)} KB)
                </p>
              ) : (
                <p style={{ color: "#94a3b8" }}>
                  Click to select an audio file (mp3, wav, webm, m4a, etc.)
                </p>
              )}
            </div>

            <button
              onClick={handleUploadAnalyze}
              disabled={!uploadFile || uploadAnalyzing}
              style={!uploadFile || uploadAnalyzing ? btnDisabled : btnPrimary}
            >
              {uploadAnalyzing ? "Analyzing..." : "Upload & Analyze"}
            </button>

            {uploadStatus && (
              <p style={{ marginTop: 12, fontSize: 14, color: "#475569" }}>
                {uploadStatus}
              </p>
            )}
          </div>

          {uploadTranscript && (
            <div style={{ ...sectionStyle, marginBottom: 16 }}>
              <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>
                Transcript
              </h3>
              <p style={{ fontSize: 14, color: "#334155", lineHeight: 1.8 }}>
                {uploadTranscript}
              </p>
            </div>
          )}

          {uploadMistakes.length > 0 && (
            <div>
              <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 12 }}>
                Analysis Results
              </h2>
              {uploadMistakes.map((m) => (
                <MistakeCard key={m.id} mistake={m} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
