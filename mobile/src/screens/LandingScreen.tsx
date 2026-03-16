import React, { memo, useCallback, useEffect, useMemo, useRef } from "react";
import { ScrollView, Text, StyleSheet, TouchableOpacity, View } from "react-native";
import * as DocumentPicker from "expo-document-picker";
import Screen from "../components/Screen";
import SectionCard from "../components/SectionCard";
import PrimaryButton from "../components/PrimaryButton";
import TopicPicker from "../components/TopicPicker";
import TranscriptEditor from "../components/TranscriptEditor";
import AudioRecorder from "../components/AudioRecorder";
import MistakeCard from "../components/MistakeCard";
import { useLanguageContext } from "../contexts/LanguageContext";
import { useLandingState } from "../contexts/LandingStateContext";
import {
  analyzeSession,
  createSessionWithAudio,
  createSessionWithTranscript,
  getSession,
  getTopicHistory,
  getTopics,
} from "../services/api";
import type { PracticeSelection, SessionDetailOut } from "../types";
import { colors } from "../theme";

function formatDuration(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60)
    .toString()
    .padStart(2, "0");
  const seconds = (totalSeconds % 60).toString().padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function LandingScreen() {
  const scrollRef = useRef<ScrollView | null>(null);
  const currentScrollYRef = useRef(0);
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
    recordedFile,
    setRecordedFile,
    bumpDataRefreshVersion,
  } = useLandingState();
  const [pendingOverrideSessionId, setPendingOverrideSessionId] = React.useState<number | null>(null);
  const [pendingOverrideScope, setPendingOverrideScope] = React.useState<"record" | "upload" | null>(null);

  const selectedTopic = useMemo(
    () => topics.find((topic) => topic.key === selectedTopicKey) ?? null,
    [topics, selectedTopicKey]
  );

  useEffect(() => {
    if (!currentLanguageProfile) {
      return;
    }
    let cancelled = false;
    void (async () => {
      setTopicsLoading(true);
      try {
        const data = await getTopics(currentLanguageProfile.language_code);
        if (cancelled) {
          return;
        }
        setTopics(data.topics);
        setEstimatedLevel(data.estimated_level);
        if (!data.topics.some((topic) => topic.key === selectedTopicKey)) {
          setSelectedTopicKey(data.topics[0]?.key ?? "free_talk");
        }
      } catch {
        if (!cancelled) {
          setTopics([]);
          setEstimatedLevel("beginner");
          setSelectedTopicKey("free_talk");
        }
      } finally {
        if (!cancelled) {
          setTopicsLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [currentLanguageProfile]);

  useEffect(() => {
    if (!currentLanguageProfile) {
      return;
    }
    let cancelled = false;
    void (async () => {
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
        if (!cancelled) {
          setTopicHistory([]);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [currentLanguageProfile, selectedTopicKey]);

  useEffect(() => {
    if (!isRecording) {
      return;
    }
    const timer = setInterval(() => {
      setElapsedSec((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, [isRecording]);

  const durationHint = useMemo(() => {
    if (!isRecording) {
      return recordedFile
        ? "Recording complete. Upload it for transcription and analysis."
        : "";
    }
    if (elapsedSec < 30) {
      return "Keep speaking. Target at least 30 seconds.";
    }
    if (elapsedSec <= 60) {
      return "Great length. You are in the 30-60 second target range.";
    }
    return "You can stop anytime. You already exceeded the target range.";
  }, [elapsedSec, isRecording, recordedFile]);

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

  const isLanguageMismatchMessage = (message?: string) =>
    !!message && /language mismatch/i.test(message);

  const applyRecordSessionResult = (session: SessionDetailOut) => {
    const transcript = session.transcript?.raw_text || liveTranscript.trim();
    setLiveTranscript(transcript);
    setMistakes(session.mistakes || []);
    setTranscriptAnalyzed(true);
    if (session.status === "error" && transcript) {
      setPendingOverrideSessionId(session.id);
      setPendingOverrideScope("record");
      setStatusMsg(
        "Language mismatch detected. Switch profiles and keep your transcript, or tap Analyze Anyway to continue with this profile."
      );
      return;
    }
    setPendingOverrideSessionId(null);
    setPendingOverrideScope(null);
    setStatusMsg(
      session.mistakes.length > 0
        ? `Found ${session.mistakes.length} mistake(s).`
        : "No mistakes found. This session will still appear in History and Insights."
    );
  };

  const applyUploadSessionResult = (session: SessionDetailOut) => {
    const transcript = session.transcript?.raw_text || "";
    setUploadTranscript(transcript);
    setUploadMistakes(session.mistakes || []);
    if (session.status === "error" && transcript) {
      setPendingOverrideSessionId(session.id);
      setPendingOverrideScope("upload");
      setUploadStatus(
        "Language mismatch detected. Switch profiles and re-run, or tap Analyze Anyway to continue with this profile."
      );
      return;
    }
    setPendingOverrideSessionId(null);
    setPendingOverrideScope(null);
    setUploadStatus(
      session.mistakes.length > 0
        ? `Found ${session.mistakes.length} mistake(s).`
        : "No mistakes found. This session will still appear in History and Insights."
    );
  };

  const handleAnalyzeRecordedAudio = async () => {
    if (!currentLanguageProfile) {
      setStatusMsg("Select a language profile first.");
      return;
    }
    const trimmedTranscript = liveTranscript.trim();
    const shouldAnalyzeTypedTranscript = trimmedTranscript.length > 0;
    if (!recordedFile && !shouldAnalyzeTypedTranscript) {
      setStatusMsg("Record audio first, or type transcript text before analyzing.");
      return;
    }
    setAnalyzing(true);
    try {
      if (!shouldAnalyzeTypedTranscript && recordedFile) {
        setStatusMsg("Uploading recorded audio...");
        const session = await createSessionWithAudio(
          recordedFile,
          currentLanguageProfile.id,
          buildPracticeSelection()
        );
        const detail = await getSession(session.id);
        applyRecordSessionResult(detail);
        bumpDataRefreshVersion();
      } else {
        setStatusMsg("Saving session...");
        const session = await createSessionWithTranscript(
          trimmedTranscript,
          currentLanguageProfile.id,
          buildPracticeSelection()
        );
        setStatusMsg("Analyzing...");
        try {
          await analyzeSession(session.id);
          const detail = await getSession(session.id);
          applyRecordSessionResult(detail);
          bumpDataRefreshVersion();
        } catch (err: any) {
          if (isLanguageMismatchMessage(err?.message)) {
            setPendingOverrideSessionId(session.id);
            setPendingOverrideScope("record");
            setTranscriptAnalyzed(false);
            setStatusMsg(
              "Language mismatch detected. Switch profiles and keep your transcript, or tap Analyze Anyway to continue with this profile."
            );
          } else {
            throw err;
          }
        }
      }
    } catch (err: any) {
      setStatusMsg(err?.message || "Analysis failed.");
    } finally {
      setAnalyzing(false);
    }
  };

  const handlePickAudio = async () => {
    const result = await DocumentPicker.getDocumentAsync({
      type: ["audio/*"],
      copyToCacheDirectory: true,
    });
    if (result.canceled || !result.assets?.[0]) {
      return;
    }
    const asset = result.assets[0];
    setUploadFile({
      uri: asset.uri,
      name: asset.name,
      type: asset.mimeType || "audio/m4a",
    });
  };

  const handleUploadAnalyze = async () => {
    if (!currentLanguageProfile) {
      setUploadStatus("Select a language profile first.");
      return;
    }
    if (!uploadFile) {
      return;
    }
    setUploadAnalyzing(true);
    setUploadStatus("Uploading and analyzing...");
    try {
      const session = await createSessionWithAudio(
        uploadFile,
        currentLanguageProfile.id,
        buildPracticeSelection()
      );
      const detail = await getSession(session.id);
      applyUploadSessionResult(detail);
      bumpDataRefreshVersion();
    } catch (err: any) {
      setUploadStatus(err?.message || "Upload analysis failed.");
    } finally {
      setUploadAnalyzing(false);
    }
  };

  const handleReset = () => {
    setLiveTranscript("");
    setTranscriptAnalyzed(false);
    setMistakes([]);
    setStatusMsg("");
    setElapsedSec(0);
    setRecordedFile(null);
    setPendingOverrideSessionId(null);
    setPendingOverrideScope(null);
  };

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

  const handleOverrideAnalyze = async () => {
    if (!pendingOverrideSessionId) {
      return;
    }
    if (pendingOverrideScope === "upload") {
      setUploadAnalyzing(true);
      setUploadStatus("Analyzing anyway...");
    } else {
      setAnalyzing(true);
      setStatusMsg("Analyzing anyway...");
    }
    try {
      await analyzeSession(pendingOverrideSessionId, undefined, true);
      const detail = await getSession(pendingOverrideSessionId);
      if (pendingOverrideScope === "upload") {
        applyUploadSessionResult(detail);
      } else {
        applyRecordSessionResult(detail);
      }
      bumpDataRefreshVersion();
    } catch (err: any) {
      if (pendingOverrideScope === "upload") {
        setUploadStatus(err?.message || "Override analysis failed.");
      } else {
        setStatusMsg(err?.message || "Override analysis failed.");
      }
    } finally {
      setAnalyzing(false);
      setUploadAnalyzing(false);
    }
  };

  const handleRandomizeTopic = () => {
    const selectable = topics.filter((topic) => topic.key !== "free_talk");
    if (selectable.length === 0) {
      return;
    }
    const next = selectable[Math.floor(Math.random() * selectable.length)];
    setSelectedTopicKey(next.key);
  };

  return (
    <Screen
      ref={scrollRef}
      onScroll={(event) => {
        currentScrollYRef.current = event.nativeEvent.contentOffset.y;
      }}
    >
      <Text style={styles.title}>Language Tutor</Text>
      <Text style={styles.subtitle}>
        Practice with a suggested topic or choose Free Talk, then get analysis and
        track improvement over time.
      </Text>

      {isLoadingLanguage ? (
        <SectionCard>
          <Text style={styles.muted}>Loading language profiles...</Text>
        </SectionCard>
      ) : currentLanguageProfile ? (
        <SectionCard>
          <Text style={styles.meta}>
            Estimated level: <Text style={styles.metaStrong}>{estimatedLevel}</Text>
          </Text>
          <View style={styles.topicRow}>
            <View style={styles.topicPickerWrap}>
              {topicsLoading ? (
                <Text style={styles.muted}>Loading topics...</Text>
              ) : (
                <TopicPicker
                  topics={topics}
                  selectedTopicKey={selectedTopicKey}
                  onSelect={setSelectedTopicKey}
                />
              )}
            </View>
            <PrimaryButton
              label="Randomize"
              onPress={handleRandomizeTopic}
              disabled={topics.length <= 1}
              style={styles.compactButton}
            />
          </View>
          {selectedTopic ? (
            <Text style={styles.prompt}>
              <Text style={styles.metaStrong}>Prompt:</Text> {selectedTopic.prompt}
            </Text>
          ) : null}
          {selectedTopicKey !== "free_talk" && topicHistory.length > 0 ? (
            <View style={styles.historyBox}>
              <Text style={styles.historyTitle}>Previous attempts for this topic</Text>
              {topicHistory.slice(0, 3).map((attempt) => (
                <Text key={attempt.id} style={styles.historyRow}>
                  {attempt.date}  |  Mistakes: {attempt.mistake_count}
                </Text>
              ))}
            </View>
          ) : null}
        </SectionCard>
      ) : (
        <SectionCard>
          <Text style={styles.muted}>
            No language profile found yet. A default profile should appear after
            backend restart.
          </Text>
        </SectionCard>
      )}

      {currentLanguageProfile ? (
        <View style={styles.segmented}>
          <TouchableOpacity
            style={[styles.segment, tab === "record" && styles.segmentActive]}
            onPress={() => setTab("record")}
          >
            <Text
              style={[
                styles.segmentText,
                tab === "record" && styles.segmentTextActive,
              ]}
            >
              Record Audio
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.segment, tab === "upload" && styles.segmentActive]}
            onPress={() => setTab("upload")}
          >
            <Text
              style={[
                styles.segmentText,
                tab === "upload" && styles.segmentTextActive,
              ]}
            >
              Upload Audio
            </Text>
          </TouchableOpacity>
        </View>
      ) : null}

      {tab === "record" && currentLanguageProfile ? (
        <>
          <SectionCard>
            <View style={styles.actionRow}>
              <AudioRecorder
                onStatusChange={(recording) => {
                  setIsRecording(recording);
                  if (recording) {
                    setElapsedSec(0);
                    setStatusMsg("");
                    setTranscriptAnalyzed(false);
                    setMistakes([]);
                  }
                }}
                onRecordingReady={(file) => {
                  setRecordedFile(file);
                  if (file) {
                    setStatusMsg("Recording ready to upload.");
                  }
                }}
              />
              <View>
                <Text style={styles.timer}>Timer {formatDuration(elapsedSec)}</Text>
                {durationHint ? <Text style={styles.hint}>{durationHint}</Text> : null}
              </View>
            </View>

            <Text style={styles.note}>
              Mobile currently records locally, then uploads the finished file for
              backend transcription. Or type a transcript below and analyze that
              directly.
            </Text>

            <View style={styles.buttonStack}>
              <PrimaryButton
                label={recordedFile ? "Analyze Recorded Audio" : "Analyze"}
                onPress={() => {
                  void handleAnalyzeRecordedAudio();
                }}
                disabled={(!recordedFile && !liveTranscript.trim()) || isRecording}
                loading={analyzing}
                tone="success"
              />
              <PrimaryButton
                label="Reset"
                onPress={handleReset}
                disabled={isRecording}
                tone="neutral"
              />
            </View>

            <TranscriptEditor
              transcript={liveTranscript}
              editable={!isRecording}
              onFocus={() => {
                setTimeout(() => {
                  scrollRef.current?.scrollTo({
                    y: currentScrollYRef.current + 96,
                    animated: true,
                  });
                }, 120);
              }}
              onChangeText={(value) => {
                setLiveTranscript(value);
                setTranscriptAnalyzed(false);
                setPendingOverrideSessionId(null);
                setPendingOverrideScope(null);
                if (isLanguageMismatchMessage(statusMsg)) {
                  setStatusMsg("");
                }
                if (value.trim()) {
                  setRecordedFile(null);
                }
              }}
            />

            {statusMsg ? <Text style={styles.status}>{statusMsg}</Text> : null}
            {pendingOverrideSessionId && pendingOverrideScope === "record" ? (
              <PrimaryButton
                label="Analyze Anyway"
                onPress={() => {
                  void handleOverrideAnalyze();
                }}
                disabled={analyzing || isRecording}
                tone="neutral"
              />
            ) : null}
          </SectionCard>

          {transcriptAnalyzed && liveTranscript ? (
            <SectionCard>
              <Text style={styles.sectionTitle}>Transcript</Text>
              <Text style={styles.transcriptText}>{liveTranscript}</Text>
            </SectionCard>
          ) : null}

          {mistakes.length > 0 ? (
            <View>
              <Text style={styles.sectionTitle}>Analysis Results</Text>
              {mistakes.map((mistake) => (
                <MistakeCard key={mistake.id} mistake={mistake} />
              ))}
            </View>
          ) : null}
        </>
      ) : null}

      {tab === "upload" && currentLanguageProfile ? (
        <>
          <SectionCard>
            <TouchableOpacity style={styles.uploadDrop} onPress={() => void handlePickAudio()}>
              <Text style={styles.uploadTitle}>
                {uploadFile ? uploadFile.name : "Pick an audio file"}
              </Text>
              <Text style={styles.muted}>
                {uploadFile
                  ? uploadFile.type
                  : "mp3, wav, webm, m4a and other audio formats"}
              </Text>
            </TouchableOpacity>
            <PrimaryButton
              label="Upload and Analyze"
              onPress={() => {
                void handleUploadAnalyze();
              }}
              disabled={!uploadFile}
              loading={uploadAnalyzing}
              tone="success"
            />
            {uploadStatus ? <Text style={styles.status}>{uploadStatus}</Text> : null}
            {pendingOverrideSessionId && pendingOverrideScope === "upload" ? (
              <PrimaryButton
                label="Analyze Anyway"
                onPress={() => {
                  void handleOverrideAnalyze();
                }}
                disabled={uploadAnalyzing}
                tone="neutral"
              />
            ) : null}
          </SectionCard>

          {uploadTranscript ? (
            <SectionCard>
              <Text style={styles.sectionTitle}>Transcript</Text>
              <Text style={styles.transcriptText}>{uploadTranscript}</Text>
            </SectionCard>
          ) : null}

          {uploadMistakes.length > 0 ? (
            <View>
              <Text style={styles.sectionTitle}>Analysis Results</Text>
              {uploadMistakes.map((mistake) => (
                <MistakeCard key={mistake.id} mistake={mistake} />
              ))}
            </View>
          ) : null}
        </>
      ) : null}
    </Screen>
  );
}

export default memo(LandingScreen);

const styles = StyleSheet.create({
  title: {
    fontSize: 28,
    fontWeight: "800",
    color: colors.text,
    marginBottom: 8,
  },
  subtitle: {
    color: colors.textMuted,
    fontSize: 15,
    lineHeight: 22,
    marginBottom: 16,
  },
  meta: {
    color: colors.textMuted,
    marginBottom: 12,
  },
  metaStrong: {
    color: colors.text,
    fontWeight: "800",
  },
  label: {
    color: colors.text,
    fontSize: 14,
    fontWeight: "700",
  },
  prompt: {
    color: colors.textMuted,
    marginTop: 12,
    lineHeight: 22,
  },
  historyBox: {
    marginTop: 12,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surfaceMuted,
    padding: 12,
    gap: 4,
  },
  historyTitle: {
    color: colors.text,
    fontWeight: "700",
    fontSize: 13,
  },
  historyRow: {
    color: colors.textMuted,
    fontSize: 13,
  },
  segmented: {
    flexDirection: "row",
    backgroundColor: colors.surface,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border,
    padding: 4,
    marginBottom: 16,
  },
  segment: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 10,
    alignItems: "center",
  },
  segmentActive: {
    backgroundColor: colors.primary,
  },
  segmentText: {
    color: colors.textMuted,
    fontWeight: "700",
    fontSize: 14,
  },
  segmentTextActive: {
    color: colors.white,
  },
  rowWrap: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 12,
    marginBottom: 12,
  },
  topicRow: {
    flexDirection: "row",
    alignItems: "flex-end",
    gap: 12,
    marginBottom: 12,
  },
  topicPickerWrap: {
    flex: 1,
  },
  compactButton: {
    minWidth: 110,
  },
  actionRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 12,
    alignItems: "center",
    marginBottom: 12,
  },
  timer: {
    color: colors.text,
    fontWeight: "700",
    fontSize: 14,
  },
  hint: {
    color: colors.textSoft,
    fontSize: 12,
    marginTop: 4,
    maxWidth: 220,
  },
  note: {
    color: colors.textSoft,
    fontSize: 13,
    lineHeight: 20,
    marginBottom: 12,
  },
  buttonStack: {
    gap: 10,
    marginBottom: 14,
  },
  status: {
    color: colors.textMuted,
    marginTop: 12,
    lineHeight: 20,
  },
  sectionTitle: {
    color: colors.text,
    fontWeight: "800",
    fontSize: 18,
    marginBottom: 12,
  },
  transcriptText: {
    color: colors.textMuted,
    lineHeight: 22,
    fontSize: 15,
  },
  muted: {
    color: colors.textSoft,
    fontSize: 14,
    lineHeight: 20,
  },
  uploadDrop: {
    borderWidth: 2,
    borderStyle: "dashed",
    borderColor: colors.border,
    borderRadius: 12,
    backgroundColor: colors.surfaceMuted,
    padding: 24,
    alignItems: "center",
    marginBottom: 14,
  },
  uploadTitle: {
    color: colors.text,
    fontWeight: "700",
    fontSize: 15,
    marginBottom: 6,
  },
});
