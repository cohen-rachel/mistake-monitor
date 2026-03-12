import React, { useEffect, useState } from "react";
import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import Screen from "../components/Screen";
import SectionCard from "../components/SectionCard";
import MistakeCard from "../components/MistakeCard";
import { useLanguageContext } from "../contexts/LanguageContext";
import { getSession, listSessions } from "../services/api";
import type { SessionDetailOut, SessionOut } from "../types";
import { colors } from "../theme";

export default function HistoryScreen() {
  const { currentLanguageProfile, isLoadingLanguage } = useLanguageContext();
  const [sessions, setSessions] = useState<SessionOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [detail, setDetail] = useState<SessionDetailOut | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  useEffect(() => {
    if (!currentLanguageProfile) {
      if (!isLoadingLanguage) {
        setLoading(false);
      }
      return;
    }
    setLoading(true);
    listSessions(currentLanguageProfile.id)
      .then((data) => setSessions(data.sessions))
      .catch(() => setSessions([]))
      .finally(() => setLoading(false));
  }, [currentLanguageProfile, isLoadingLanguage]);

  const handleExpand = async (id: number) => {
    if (expandedId === id) {
      setExpandedId(null);
      setDetail(null);
      return;
    }
    setExpandedId(id);
    setDetailLoading(true);
    try {
      const loaded = await getSession(id);
      setDetail(loaded);
    } catch {
      setDetail(null);
    } finally {
      setDetailLoading(false);
    }
  };

  return (
    <Screen>
      <Text style={styles.title}>Session History</Text>
      {loading || isLoadingLanguage ? (
        <SectionCard>
          <Text style={styles.muted}>Loading...</Text>
        </SectionCard>
      ) : !currentLanguageProfile ? (
        <SectionCard>
          <Text style={styles.muted}>
            Please select or create a language profile to view history.
          </Text>
        </SectionCard>
      ) : sessions.length === 0 ? (
        <SectionCard>
          <Text style={styles.muted}>
            No sessions yet. Record or upload audio from the home screen first.
          </Text>
        </SectionCard>
      ) : (
        <View>
          {sessions.map((session) => {
            const expanded = session.id === expandedId;
            return (
              <View key={session.id}>
                <TouchableOpacity
                  style={styles.row}
                  onPress={() => {
                    void handleExpand(session.id);
                  }}
                >
                  <View style={styles.rowTop}>
                    <Text style={styles.rowTitle}>
                      {new Date(session.created_at).toLocaleString()}
                    </Text>
                    <View style={styles.badge}>
                      <Text style={styles.badgeText}>{session.status}</Text>
                    </View>
                  </View>
                  <Text style={styles.rowMeta}>Language: {session.language}</Text>
                </TouchableOpacity>
                {expanded ? (
                  <SectionCard>
                    {detailLoading ? (
                      <Text style={styles.muted}>Loading details...</Text>
                    ) : detail ? (
                      <>
                        {detail.transcript ? (
                          <>
                            <Text style={styles.sectionTitle}>Transcript</Text>
                            <Text style={styles.transcript}>
                              {detail.transcript.raw_text}
                            </Text>
                          </>
                        ) : null}
                        {detail.mistakes.length > 0 ? (
                          <>
                            <Text style={styles.sectionTitle}>
                              Mistakes ({detail.mistakes.length})
                            </Text>
                            {detail.mistakes.map((mistake) => (
                              <MistakeCard key={mistake.id} mistake={mistake} />
                            ))}
                          </>
                        ) : (
                          <Text style={styles.muted}>
                            No mistakes found for this session.
                          </Text>
                        )}
                      </>
                    ) : (
                      <Text style={styles.muted}>Could not load session details.</Text>
                    )}
                  </SectionCard>
                ) : null}
              </View>
            );
          })}
        </View>
      )}
    </Screen>
  );
}

const styles = StyleSheet.create({
  title: {
    color: colors.text,
    fontSize: 28,
    fontWeight: "800",
    marginBottom: 16,
  },
  row: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 12,
    padding: 16,
    marginBottom: 10,
  },
  rowTop: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 12,
    alignItems: "center",
    marginBottom: 8,
  },
  rowTitle: {
    color: colors.text,
    fontWeight: "700",
    flex: 1,
  },
  rowMeta: {
    color: colors.textSoft,
  },
  badge: {
    backgroundColor: colors.blueTint,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 999,
  },
  badgeText: {
    color: "#0369a1",
    fontWeight: "700",
    fontSize: 12,
  },
  sectionTitle: {
    color: colors.text,
    fontWeight: "800",
    fontSize: 17,
    marginBottom: 10,
  },
  transcript: {
    color: colors.textMuted,
    lineHeight: 22,
    fontSize: 15,
    marginBottom: 16,
  },
  muted: {
    color: colors.textSoft,
    fontSize: 14,
  },
});
