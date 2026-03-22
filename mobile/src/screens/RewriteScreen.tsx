import React, { memo, useEffect, useState } from "react";
import { StyleSheet, Text, TextInput, View } from "react-native";
import Screen from "../components/Screen";
import SectionCard from "../components/SectionCard";
import PrimaryButton from "../components/PrimaryButton";
import {
  getRewriteExercise,
  getRewriteStats,
  submitRewriteExercise,
} from "../services/api";
import { useLanguageContext } from "../contexts/LanguageContext";
import { useLandingState } from "../contexts/LandingStateContext";
import type {
  RewriteExerciseResponse,
  RewriteStatsResponse,
} from "../types";
import { colors } from "../theme";

function RewriteScreen() {
  const { currentLanguageProfile, isLoadingLanguage } = useLanguageContext();
  const { dataRefreshVersion } = useLandingState();
  const [exercise, setExercise] = useState<RewriteExerciseResponse | null>(null);
  const [stats, setStats] = useState<RewriteStatsResponse | null>(null);
  const [answer, setAnswer] = useState("");
  const [feedback, setFeedback] = useState<string | null>(null);
  const [skipMessage, setSkipMessage] = useState("");
  const [noMistakesMessage, setNoMistakesMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [seenMistakeIds, setSeenMistakeIds] = useState<number[]>([]);
  const [hasSubmitted, setHasSubmitted] = useState(false);
  const [showExplanation, setShowExplanation] = useState(false);

  const loadStats = async () => {
    if (!currentLanguageProfile) {
      return;
    }
    try {
      const loaded = await getRewriteStats(currentLanguageProfile.language_code);
      setStats(loaded);
    } catch {
      setStats(null);
    }
  };

  const loadExercise = async (options?: {
    postMessage?: string;
    excludeIds?: number[];
  }) => {
    if (!currentLanguageProfile) {
      return;
    }
    setLoading(true);
    setFeedback(null);
    setSkipMessage("");
    setNoMistakesMessage(null);
    setAnswer("");
    setHasSubmitted(false);
    setShowExplanation(false);
    try {
      const loaded = await getRewriteExercise(
        currentLanguageProfile.language_code,
        1,
        options?.excludeIds
      );
      setExercise(loaded);
      if (!seenMistakeIds.includes(loaded.source_mistake_id)) {
        setSeenMistakeIds((prev) => [...prev, loaded.source_mistake_id]);
      }
      if (options?.postMessage) {
        setSkipMessage(options.postMessage);
      }
    } catch (err: any) {
      setExercise(null);
      const message =
        err?.message && /no mistakes/i.test(err.message)
          ? "No mistakes to correct at this time."
          : null;
      setNoMistakesMessage(message);
      if (message) {
        setSeenMistakeIds([]);
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!currentLanguageProfile) {
      return;
    }
    void loadExercise();
    void loadStats();
  }, [currentLanguageProfile, dataRefreshVersion]);

  const handleSubmit = async () => {
    if (!exercise || !answer.trim() || !currentLanguageProfile) {
      return;
    }
    setLoading(true);
    try {
      const result = await submitRewriteExercise({
        user_id: 1,
        language_code: currentLanguageProfile.language_code,
        source_mistake_id: exercise.source_mistake_id,
        original_sentence: exercise.original_sentence,
        wrong_span: exercise.wrong_span,
        expected_correction: exercise.expected_correction,
        user_rewrite: answer.trim(),
      });
      const expected = result.expected_correction
        ? ` Expected correction: ${result.expected_correction}`
        : "";
      setFeedback(
        `${result.is_correct ? "Correct." : "Not quite."} Score ${Math.round(
          result.score * 100
        )}%. ${result.feedback}${expected}`
      );
      await loadStats();
      setHasSubmitted(true);
    } catch (err: any) {
      setFeedback(err?.message || "Could not submit rewrite.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Screen>
      <Text style={styles.title}>Rewrite Practice</Text>
      {isLoadingLanguage ? (
        <SectionCard>
          <Text style={styles.muted}>Loading language profile...</Text>
        </SectionCard>
      ) : !currentLanguageProfile ? (
        <SectionCard>
          <Text style={styles.muted}>
            Please select or create a language profile to start rewriting exercises.
          </Text>
        </SectionCard>
      ) : (
        <>
          <Text style={styles.subtitle}>
            You get your original incorrect sentence and rewrite it correctly.
          </Text>
          <SectionCard>
            <Text style={styles.sectionTitle}>Exercise</Text>
            {loading && !exercise ? (
              <Text style={styles.muted}>Loading exercise...</Text>
            ) : exercise ? (
              <>
                <Text style={styles.meta}>{exercise.mistake_type_label}</Text>
                <Text style={styles.original}>{exercise.original_sentence}</Text>
                <TextInput
                  value={answer}
                  onChangeText={setAnswer}
                  placeholder="Rewrite the sentence correctly..."
                  placeholderTextColor={colors.textSoft}
                  multiline
                  style={styles.input}
                />
                {exercise.explanation_short ? (
                  <View style={styles.explanationWrap}>
                    <Text
                      style={styles.explanationToggle}
                      onPress={() => setShowExplanation((prev) => !prev)}
                    >
                      {showExplanation ? "Hide explanation" : "Having trouble? Explain my error"}
                    </Text>
                    {showExplanation ? (
                      <Text style={styles.explanationText}>{exercise.explanation_short}</Text>
                    ) : null}
                  </View>
                ) : null}
                <View style={styles.buttonStack}>
                  <PrimaryButton
                    label={hasSubmitted ? "Next Sentence" : "Submit Rewrite"}
                    onPress={() => {
                      if (hasSubmitted) {
                        void loadExercise({
                          postMessage: "Next sentence loaded.",
                          excludeIds: [...seenMistakeIds],
                        });
                        return;
                      }
                      void handleSubmit();
                    }}
                    disabled={!hasSubmitted && !answer.trim()}
                    loading={loading}
                  />
                  {!hasSubmitted ? (
                    <PrimaryButton
                      label="Skip"
                      onPress={() => {
                        void loadExercise({
                          postMessage: "Skipped to the next mistake.",
                          excludeIds: [...seenMistakeIds],
                        });
                      }}
                      tone="neutral"
                      disabled={loading}
                    />
                  ) : null}
                </View>
                {feedback || skipMessage ? (
                  <Text style={styles.feedback}>{feedback || skipMessage}</Text>
                ) : null}
              </>
            ) : (
              <Text style={styles.muted}>
                {noMistakesMessage ||
                  "No rewrite exercises available yet. Only sessions with detected mistakes appear here."}
              </Text>
            )}
          </SectionCard>

          <SectionCard>
            <Text style={styles.sectionTitle}>Rewrite Stats</Text>
            {loading && !stats ? (
              <Text style={styles.muted}>Loading stats...</Text>
            ) : !stats ? (
              <Text style={styles.muted}>No stats yet.</Text>
            ) : (
              <View style={styles.statsCard}>
                <Text style={styles.statsValue}>
                  {Math.round(stats.overall_accuracy * 100)}%
                </Text>
                <Text style={styles.statsLabel}>Rewrite accuracy</Text>
                <Text style={styles.statsMeta}>
                  {stats.total_correct} correct out of {stats.total_attempts} attempts
                </Text>
              </View>
            )}
          </SectionCard>
        </>
      )}
    </Screen>
  );
}

export default memo(RewriteScreen);

const styles = StyleSheet.create({
  title: {
    color: colors.text,
    fontSize: 28,
    fontWeight: "800",
    marginBottom: 8,
  },
  subtitle: {
    color: colors.textMuted,
    marginBottom: 16,
    fontSize: 15,
    lineHeight: 22,
  },
  sectionTitle: {
    color: colors.text,
    fontSize: 18,
    fontWeight: "800",
    marginBottom: 12,
  },
  meta: {
    color: colors.textSoft,
    marginBottom: 8,
  },
  original: {
    color: colors.text,
    fontSize: 16,
    lineHeight: 24,
    marginBottom: 10,
  },
  explanationWrap: {
    marginBottom: 12,
  },
  explanationToggle: {
    color: colors.primary,
    fontSize: 12,
    fontWeight: "700",
    marginBottom: 10,
  },
  explanationText: {
    color: colors.textMuted,
    lineHeight: 22,
    backgroundColor: colors.surfaceMuted,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 12,
    padding: 12,
  },
  input: {
    minHeight: 96,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 12,
    padding: 12,
    fontSize: 15,
    color: colors.text,
    textAlignVertical: "top",
    marginBottom: 12,
  },
  buttonStack: {
    gap: 10,
  },
  feedback: {
    marginTop: 12,
    color: colors.textMuted,
    lineHeight: 20,
  },
  statsCard: {
    backgroundColor: colors.surfaceMuted,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 14,
    paddingVertical: 22,
    paddingHorizontal: 16,
    alignItems: "center",
  },
  statsValue: {
    color: "#4338ca",
    fontSize: 40,
    fontWeight: "800",
    marginBottom: 8,
  },
  statsLabel: {
    color: colors.text,
    fontSize: 16,
    fontWeight: "700",
    marginBottom: 4,
  },
  statsMeta: {
    color: colors.textSoft,
    fontSize: 14,
  },
  muted: {
    color: colors.textSoft,
    fontSize: 14,
  },
});
