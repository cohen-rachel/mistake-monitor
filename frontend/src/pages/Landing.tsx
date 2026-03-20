import { useCallback, useRef, useEffect, useMemo, useState } from "react";
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
import { useLandingState } from "../contexts/LandingStateContext";

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

// Keep the batching scaffolding in place, but disable timed auto-analysis for now.
const ENABLE_AUTO_ANALYZE_BATCHING = false;
const AUTO_ANALYZE_INTERVAL_SEC = 20;

function formatDuration(totalSeconds: number): string {
  const m = Math.floor(totalSeconds / 60)
    .toString()
    .padStart(2, "0");
  const s = (totalSeconds % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

type QueuedAutoBatch = {
  batchIndex: number;
  transcript: string;
};

function isNonRetryableAutoAnalyzeError(error: unknown): boolean {
  const status = (error as { status?: number } | null)?.status;
  return typeof status === "number" && status >= 400 && status < 500;
}

export default function Landing() {
  const { currentLanguageProfile, isLoadingLanguage } = useLanguageContext();
  const {
    tab,
    setTab,
    topics,
    setTopics,
    estimatedLevel,
    setEstimatedLevel,
    selectedTopicKey,
    setSelectedTopicKey,
    topicHistory,
    setTopicHistory,
    topicsLoading,
    setTopicsLoading,
    isRecording,
    setIsRecording,
    elapsedSec,
    setElapsedSec,
    liveTranscript,
    setLiveTranscript,
    transcriptAnalyzed,
    setTranscriptAnalyzed,
    analyzing,
    setAnalyzing,
    mistakes,
    setMistakes,
    statusMsg,
    setStatusMsg,
    uploadFile,
    setUploadFile,
    uploadAnalyzing,
    setUploadAnalyzing,
    uploadMistakes,
    setUploadMistakes,
    uploadTranscript,
    setUploadTranscript,
    uploadStatus,
    setUploadStatus,
  } = useLandingState();
  const [autoAnalyzing, setAutoAnalyzing] = useState(false);
  const [autoQueueVersion, setAutoQueueVersion] = useState(0);
  const [pendingOverrideSessionId, setPendingOverrideSessionId] = useState<number | null>(null);
  const [pendingOverrideScope, setPendingOverrideScope] = useState<"record" | "upload" | null>(null);
  const [provisionalTranscript, setProvisionalTranscript] = useState("");
  const [finalTranscript, setFinalTranscript] = useState("");
  const [finalizingTranscript, setFinalizingTranscript] = useState(false);
  const [userEditedTranscript, setUserEditedTranscript] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const liveTranscriptRef = useRef(liveTranscript);
  const provisionalTranscriptRef = useRef("");
  const finalTranscriptRef = useRef("");
  const userEditedTranscriptRef = useRef(false);
  const recordingBaseTranscriptRef = useRef("");
  const recordingSessionFullTranscriptRef = useRef("");
  const pendingAutoTranscriptRef = useRef("");
  const lastAutoBatchIndexRef = useRef(0);
  const autoBatchQueueRef = useRef<QueuedAutoBatch[]>([]);
  const autoQueueRunningRef = useRef(false);

  const selectedTopic = useMemo(
    () => topics.find((t) => t.key === selectedTopicKey) ?? null,
    [topics, selectedTopicKey]
  );

  useEffect(() => {
    liveTranscriptRef.current = liveTranscript;
  }, [liveTranscript]);

  useEffect(() => {
    provisionalTranscriptRef.current = provisionalTranscript;
  }, [provisionalTranscript]);

  useEffect(() => {
    finalTranscriptRef.current = finalTranscript;
  }, [finalTranscript]);

  useEffect(() => {
    userEditedTranscriptRef.current = userEditedTranscript;
  }, [userEditedTranscript]);

  useEffect(() => {
    setPendingOverrideSessionId(null);
    setPendingOverrideScope(null);
    if (isLanguageMismatchMessage(statusMsg)) {
      setStatusMsg("");
    }
    if (isLanguageMismatchMessage(uploadStatus)) {
      setUploadStatus("");
    }
  }, [currentLanguageProfile?.id]);

  const isLanguageMismatchMessage = (message?: string) =>
    !!message && /language mismatch/i.test(message);

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

  const analyzeTranscript = useCallback(
    async (transcript: string, options?: { auto?: boolean }) => {
      if (!currentLanguageProfile) return null;
      const trimmed = transcript.trim();
      if (!trimmed) return null;
      const isAuto = !!options?.auto;
      if (isAuto && (autoAnalyzing || analyzing)) {
        return null;
      }
      let createdSessionId: number | null = null;
      const setLoading = isAuto ? setAutoAnalyzing : setAnalyzing;
      setLoading(true);
      try {
        setStatusMsg(isAuto ? "Auto analyzing partial transcript..." : "Saving session...");
        const session = await createSessionWithTranscript(
          trimmed,
          currentLanguageProfile.id,
          buildPracticeSelection()
        );
        createdSessionId = session.id;
        if (!isAuto) {
          setStatusMsg("Analyzing...");
        }
        const result = await analyzeSession(session.id);
        setMistakes((prev) => (isAuto ? [...prev, ...result.mistakes] : result.mistakes));
        setPendingOverrideSessionId(null);
        setPendingOverrideScope(null);
        setStatusMsg(
          result.mistakes.length > 0
            ? `Found ${result.mistakes.length} mistake(s).`
            : "No mistakes found. This session will still appear in History and Insights."
        );
        if (selectedTopicKey !== "free_talk") {
          const updated = await getTopicHistory(
            selectedTopicKey,
            currentLanguageProfile.language_code
          );
          setTopicHistory(updated.attempts);
        }
        if (!isAuto) {
          setTranscriptAnalyzed(true);
        }
        return result;
      } catch (err: any) {
        if (!isAuto && isLanguageMismatchMessage(err?.message)) {
          setPendingOverrideSessionId(createdSessionId);
          setPendingOverrideScope("record");
          setStatusMsg(
            "Language mismatch detected. Switch profiles and keep your transcript, or click Analyze Anyway to continue with this profile."
          );
          return null;
        }
        setStatusMsg(`Error: ${err?.message || "analysis failed"}`);
        throw err;
      } finally {
        setLoading(false);
      }
    },
    [
      currentLanguageProfile,
      autoAnalyzing,
      analyzing,
      buildPracticeSelection,
      selectedTopicKey,
      setTopicHistory,
    ]
  );

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

  const handleChunk = useCallback(
    (text: string, fullText?: string) => {
      const nextSessionTranscript = fullText?.trim()
        ? fullText.trim()
        : `${recordingSessionFullTranscriptRef.current} ${text}`.trim();
      recordingSessionFullTranscriptRef.current = nextSessionTranscript;
      const deltaText = text.trim();
      if (deltaText) {
        pendingAutoTranscriptRef.current = [
          pendingAutoTranscriptRef.current,
          deltaText,
        ]
          .filter(Boolean)
          .join(" ")
          .trim();
      }
      const nextTranscript = [recordingBaseTranscriptRef.current, nextSessionTranscript]
        .filter(Boolean)
        .join(" ")
        .trim();
      setProvisionalTranscript(nextTranscript);
      if (!userEditedTranscriptRef.current && !finalTranscriptRef.current.trim()) {
        setLiveTranscript(nextTranscript);
      }
      setTranscriptAnalyzed(false);
    },
    [setTranscriptAnalyzed]
  );

  const handleFinalTranscript = useCallback(
    (analysisText: string, displayText: string) => {
      if (!analysisText.trim() && !displayText.trim()) {
        const authoritativeText =
          provisionalTranscriptRef.current.trim() || liveTranscriptRef.current.trim();
        setFinalTranscript(authoritativeText);
        if (!userEditedTranscriptRef.current) {
          setLiveTranscript(authoritativeText);
          setStatusMsg("Live transcript ready.");
        } else {
          setStatusMsg(
            "Using the live transcript. Your manual edits are preserved and will be used for analysis."
          );
        }
        return;
      }

      const mergedAnalysisText = [recordingBaseTranscriptRef.current, analysisText.trim()]
        .filter(Boolean)
        .join(" ")
        .trim();
      const mergedDisplayText = [recordingBaseTranscriptRef.current, displayText.trim()]
        .filter(Boolean)
        .join(" ")
        .trim();

      setFinalTranscript(mergedAnalysisText);
      if (!userEditedTranscriptRef.current) {
        setLiveTranscript(mergedDisplayText);
        setStatusMsg("Final transcript ready.");
      } else {
        setStatusMsg(
          "Final transcript ready. Your manual edits are preserved and will be used for analysis."
        );
      }
    },
    [setLiveTranscript, setStatusMsg]
  );

  const handleFinalTranscriptError = useCallback((message: string) => {
    setStatusMsg(
      `Final transcription failed. You can still analyze the current text. ${message}`
    );
  }, [setStatusMsg]);

  const handleStatusChange = useCallback((recording: boolean) => {
    setIsRecording(recording);
    if (recording) {
      setFinalTranscript("");
      setProvisionalTranscript("");
      setFinalizingTranscript(false);
      setUserEditedTranscript(false);
      recordingSessionFullTranscriptRef.current = "";
      finalTranscriptRef.current = "";
      provisionalTranscriptRef.current = "";
      userEditedTranscriptRef.current = false;
      lastAutoBatchIndexRef.current = 0;
      pendingAutoTranscriptRef.current = "";
      autoBatchQueueRef.current = [];
      autoQueueRunningRef.current = false;
      setAutoQueueVersion((prev) => prev + 1);
      setElapsedSec(0);
      setStatusMsg("");
      if (transcriptAnalyzed) {
        recordingBaseTranscriptRef.current = "";
        setMistakes([]);
        setLiveTranscript("");
        setTranscriptAnalyzed(false);
      } else {
        recordingBaseTranscriptRef.current = liveTranscriptRef.current.trim();
        pendingAutoTranscriptRef.current = liveTranscriptRef.current.trim();
      }
    } else {
      setFinalizingTranscript(true);
      setStatusMsg("Finalizing transcript...");
    }
  }, [setElapsedSec, setIsRecording, setLiveTranscript, setMistakes, setStatusMsg, setTranscriptAnalyzed, transcriptAnalyzed]);

  useEffect(() => {
    if (!ENABLE_AUTO_ANALYZE_BATCHING) return;
    if (!isRecording) return;
    const completedBatches = Math.floor(elapsedSec / AUTO_ANALYZE_INTERVAL_SEC);
    if (completedBatches <= lastAutoBatchIndexRef.current) return;

    for (
      let batchIndex = lastAutoBatchIndexRef.current + 1;
      batchIndex <= completedBatches;
      batchIndex += 1
    ) {
      const transcript = pendingAutoTranscriptRef.current.trim();
      lastAutoBatchIndexRef.current = batchIndex;
      if (!transcript) {
        continue;
      }
      pendingAutoTranscriptRef.current = "";
      autoBatchQueueRef.current.push({
        batchIndex,
        transcript,
      });
    }
    setAutoQueueVersion((prev) => prev + 1);
  }, [
    elapsedSec,
    isRecording,
  ]);

  useEffect(() => {
    if (!ENABLE_AUTO_ANALYZE_BATCHING) return;
    if (autoQueueRunningRef.current || analyzing) return;
    const nextBatch = autoBatchQueueRef.current[0];
    if (!nextBatch) return;

    let cancelled = false;
    autoQueueRunningRef.current = true;

    void (async () => {
      try {
        const result = await analyzeTranscript(nextBatch.transcript, { auto: true });
        if (!cancelled && result) {
          autoBatchQueueRef.current.shift();
          setTranscriptAnalyzed(false);
        }
      } catch (error) {
        if (!cancelled) {
          if (!isNonRetryableAutoAnalyzeError(error)) {
            pendingAutoTranscriptRef.current = [
              nextBatch.transcript,
              pendingAutoTranscriptRef.current,
            ]
              .filter(Boolean)
              .join(" ")
              .trim();
            autoBatchQueueRef.current = [];
            lastAutoBatchIndexRef.current = Math.max(nextBatch.batchIndex - 1, 0);
          } else {
            setStatusMsg("Skipped one background batch because the transcript was not analyzable.");
          }
          autoBatchQueueRef.current.shift();
        }
      } finally {
        autoQueueRunningRef.current = false;
        if (!cancelled) {
          setAutoQueueVersion((prev) => prev + 1);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [
    analyzeTranscript,
    analyzing,
    autoQueueVersion,
    setTranscriptAnalyzed,
  ]);

  const handleRandomizeTopic = () => {
    const selectable = topics.filter((t) => t.key !== "free_talk");
    if (selectable.length === 0) return;
    const next = selectable[Math.floor(Math.random() * selectable.length)];
    setSelectedTopicKey(next.key);
  };

  const handleAnalyzeRecording = useCallback(async () => {
    if (!currentLanguageProfile) {
      setStatusMsg("Select a language profile first.");
      return;
    }
    if (autoAnalyzing) {
      setStatusMsg("Waiting for background analysis to finish...");
      return;
    }
    const editedText = liveTranscript.trim();
    const hasEditedText = userEditedTranscript && editedText.length > 0;
    const chosenTranscript = hasEditedText
      ? editedText
      : finalTranscript.trim() || provisionalTranscript.trim() || editedText;

    if (finalizingTranscript && !hasEditedText) {
      setStatusMsg("Finalizing transcript. Wait a moment or edit the text and analyze that version.");
      return;
    }
    if (!chosenTranscript) {
      setStatusMsg("No transcript to analyze. Record something first.");
      return;
    }

    try {
      const result = await analyzeTranscript(chosenTranscript);
      if (!result) {
        return;
      }
    } catch {
      // Status text is already set inside analyzeTranscript.
    }
  }, [
    analyzeTranscript,
    autoAnalyzing,
    currentLanguageProfile,
    liveTranscript,
  ]);

  const handleReset = () => {
    recordingBaseTranscriptRef.current = "";
    recordingSessionFullTranscriptRef.current = "";
    provisionalTranscriptRef.current = "";
    finalTranscriptRef.current = "";
    userEditedTranscriptRef.current = false;
    pendingAutoTranscriptRef.current = "";
    lastAutoBatchIndexRef.current = 0;
    autoBatchQueueRef.current = [];
    autoQueueRunningRef.current = false;
    setAutoQueueVersion((prev) => prev + 1);
    setLiveTranscript("");
    setProvisionalTranscript("");
    setFinalTranscript("");
    setTranscriptAnalyzed(false);
    setUserEditedTranscript(false);
    setMistakes([]);
    setStatusMsg("");
    setElapsedSec(0);
    setAutoAnalyzing(false);
    setFinalizingTranscript(false);
    setPendingOverrideSessionId(null);
    setPendingOverrideScope(null);
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
      if (session.status === "error" && session.transcript?.raw_text) {
        setPendingOverrideSessionId(session.id);
        setPendingOverrideScope("upload");
        setUploadStatus(
          "Language mismatch detected. Switch profiles and re-run, or click Analyze Anyway to continue with this profile."
        );
      } else {
        setPendingOverrideSessionId(null);
        setPendingOverrideScope(null);
        setUploadStatus(
          session.mistakes.length > 0
            ? `Found ${session.mistakes.length} mistake(s).`
            : "No mistakes found. This session will still appear in History and Insights."
        );
      }
    } catch (err: any) {
      setUploadStatus(`Error: ${err.message}`);
    } finally {
      setUploadAnalyzing(false);
    }
  };

  const handleOverrideAnalyze = async () => {
    if (!pendingOverrideSessionId) return;
    const isUpload = pendingOverrideScope === "upload";
    if (isUpload) {
      setUploadAnalyzing(true);
      setUploadStatus("Analyzing anyway...");
    } else {
      setAnalyzing(true);
      setStatusMsg("Analyzing anyway...");
    }
    try {
      const result = await analyzeSession(pendingOverrideSessionId, undefined, true);
      setPendingOverrideSessionId(null);
      setPendingOverrideScope(null);
      if (isUpload) {
        setUploadMistakes(result.mistakes);
        setUploadStatus(
          result.mistakes.length > 0
            ? `Found ${result.mistakes.length} mistake(s).`
            : "No mistakes found. This session will still appear in History and Insights."
        );
      } else {
        setMistakes(result.mistakes);
        setTranscriptAnalyzed(true);
        setStatusMsg(
          result.mistakes.length > 0
            ? `Found ${result.mistakes.length} mistake(s).`
            : "No mistakes found. This session will still appear in History and Insights."
        );
      }
    } catch (err: any) {
      if (isUpload) {
        setUploadStatus(`Error: ${err.message || "override failed"}`);
      } else {
        setStatusMsg(`Error: ${err.message || "override failed"}`);
      }
    } finally {
      setAnalyzing(false);
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
                onFinalTranscript={handleFinalTranscript}
                onError={handleFinalTranscriptError}
                onFinalizeStateChange={setFinalizingTranscript}
              />
              <div style={{ fontSize: 14, color: "#475569" }}>
                Timer: <strong>{formatDuration(elapsedSec)}</strong>
              </div>
              {!isRecording && liveTranscript.length > 0 && (
                <button
                  onClick={handleAnalyzeRecording}
                  disabled={analyzing || autoAnalyzing || (finalizingTranscript && !userEditedTranscript)}
                  style={analyzing || autoAnalyzing || (finalizingTranscript && !userEditedTranscript) ? btnDisabled : btnPrimary}
                >
                  {analyzing || autoAnalyzing ? "Analyzing..." : finalizingTranscript && !userEditedTranscript ? "Finalizing..." : "Analyze"}
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

            <TranscriptDisplay
              transcript={liveTranscript}
              isRecording={isRecording}
              onTranscriptChange={(value) => {
                recordingBaseTranscriptRef.current = value.trim();
                recordingSessionFullTranscriptRef.current = "";
                finalTranscriptRef.current = "";
                pendingAutoTranscriptRef.current = value.trim();
                lastAutoBatchIndexRef.current = 0;
                autoBatchQueueRef.current = [];
                autoQueueRunningRef.current = false;
                setAutoQueueVersion((prev) => prev + 1);
                setLiveTranscript(value);
                setFinalTranscript("");
                setUserEditedTranscript(true);
                setTranscriptAnalyzed(false);
                setPendingOverrideSessionId(null);
                setPendingOverrideScope(null);
                if (isLanguageMismatchMessage(statusMsg)) {
                  setStatusMsg("");
                }
              }}
            />

            {statusMsg && (
              <p style={{ marginTop: 12, fontSize: 14, color: "#475569" }}>
                {statusMsg}
              </p>
            )}
            {pendingOverrideSessionId && pendingOverrideScope === "record" && (
              <div style={{ marginTop: 12 }}>
                <button
                  onClick={handleOverrideAnalyze}
                  disabled={analyzing || autoAnalyzing}
                  style={analyzing || autoAnalyzing ? btnDisabled : { ...btnPrimary, background: "#0f766e" }}
                >
                  Analyze Anyway
                </button>
              </div>
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
            {pendingOverrideSessionId && pendingOverrideScope === "upload" && (
              <div style={{ marginTop: 12 }}>
                <button
                  onClick={handleOverrideAnalyze}
                  disabled={uploadAnalyzing}
                  style={uploadAnalyzing ? btnDisabled : { ...btnPrimary, background: "#0f766e" }}
                >
                  Analyze Anyway
                </button>
              </div>
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
