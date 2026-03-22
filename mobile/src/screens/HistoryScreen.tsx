import React, { memo, useEffect, useMemo, useState } from "react";
import {
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import Screen from "../components/Screen";
import SectionCard from "../components/SectionCard";
import MistakeCard from "../components/MistakeCard";
import SelectField from "../components/SelectField";
import { useLanguageContext } from "../contexts/LanguageContext";
import { useLandingState } from "../contexts/LandingStateContext";
import { getSession, listSessions } from "../services/api";
import type { SessionDetailOut, SessionOut } from "../types";
import { colors } from "../theme";

type SortMode = "date" | "mistake";

const sortOptions = [
  { label: "Date", value: "date" },
  { label: "Mistake type", value: "mistake" },
];

function formatSessionBadge(session: SessionOut): {
  label: string;
  tone: "error" | "mistakes" | "clear";
} {
  if (session.status === "error") {
    return { label: "Error", tone: "error" };
  }
  if (session.mistake_count > 0) {
    return {
      label: `${session.mistake_count} ${session.mistake_count === 1 ? "mistake" : "mistakes"}`,
      tone: "mistakes",
    };
  }
  return { label: "No mistakes", tone: "clear" };
}

function parseDateInput(value: string, endOfDay: boolean = false): Date | null {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    return null;
  }
  return new Date(`${value}T${endOfDay ? "23:59:59" : "00:00:00"}`);
}

function HistoryScreen() {
  const { currentLanguageProfile, isLoadingLanguage } = useLanguageContext();
  const { dataRefreshVersion } = useLandingState();
  const [sessions, setSessions] = useState<SessionOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [detail, setDetail] = useState<SessionDetailOut | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [showMistakesOnly, setShowMistakesOnly] = useState(false);
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [sortMode, setSortMode] = useState<SortMode>("date");
  const [openGroups, setOpenGroups] = useState<string[]>([]);

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
  }, [currentLanguageProfile, isLoadingLanguage, dataRefreshVersion]);

  const filteredSessions = useMemo(() => {
    let next = [...sessions];

    if (showMistakesOnly) {
      next = next.filter((session) => session.mistake_count > 0);
    }

    const from = parseDateInput(dateFrom);
    const to = parseDateInput(dateTo, true);

    if (from) {
      next = next.filter((session) => new Date(session.created_at) >= from);
    }

    if (to) {
      next = next.filter((session) => new Date(session.created_at) <= to);
    }

    if (sortMode === "date") {
      next.sort(
        (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      );
    } else {
      next.sort((a, b) => {
        const aLabel =
          a.primary_mistake_type_label ||
          (a.mistake_count > 0 ? "Other mistakes" : "No mistakes");
        const bLabel =
          b.primary_mistake_type_label ||
          (b.mistake_count > 0 ? "Other mistakes" : "No mistakes");
        if (aLabel !== bLabel) {
          return aLabel.localeCompare(bLabel);
        }
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      });
    }

    if (!dateFrom && !dateTo) {
      next = next.slice(0, 10);
    }

    return next;
  }, [dateFrom, dateTo, sessions, showMistakesOnly, sortMode]);

  const groupedSessions = useMemo(() => {
    const groups = new Map<string, SessionOut[]>();
    for (const session of filteredSessions) {
      const label =
        session.status === "error"
          ? "Error"
          : session.mistake_count > 0
          ? session.primary_mistake_type_label || "Other mistakes"
          : "No mistakes";
      const existing = groups.get(label);
      if (existing) {
        existing.push(session);
      } else {
        groups.set(label, [session]);
      }
    }
    return Array.from(groups.entries()).sort((a, b) => a[0].localeCompare(b[0]));
  }, [filteredSessions]);

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

  const toggleGroup = (label: string) => {
    setOpenGroups((prev) =>
      prev.includes(label) ? prev.filter((item) => item !== label) : [...prev, label]
    );
  };

  const renderSessionRow = (session: SessionOut) => {
    const expanded = session.id === expandedId;
    const badge = formatSessionBadge(session);
    const badgeStyle =
      badge.tone === "error"
        ? styles.badgeError
        : badge.tone === "mistakes"
        ? styles.badgeMistakes
        : styles.badgeClear;
    const badgeTextStyle =
      badge.tone === "error"
        ? styles.badgeTextError
        : badge.tone === "mistakes"
        ? styles.badgeTextMistakes
        : styles.badgeTextClear;

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
            <View style={[styles.badge, badgeStyle]}>
              <Text style={[styles.badgeText, badgeTextStyle]}>{badge.label}</Text>
            </View>
          </View>
          <Text style={styles.rowMeta}>Language: {session.language}</Text>
          {session.primary_focus_label && session.mistake_count > 0 ? (
            <Text style={styles.focusText}>Focus: {session.primary_focus_label}</Text>
          ) : null}
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
                    <Text style={styles.transcript}>{detail.transcript.raw_text}</Text>
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
                  <Text style={styles.muted}>No mistakes found for this session.</Text>
                )}
              </>
            ) : (
              <Text style={styles.muted}>Could not load session details.</Text>
            )}
          </SectionCard>
        ) : null}
      </View>
    );
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
      ) : (
        <>
          <SectionCard>
            <Text style={styles.sectionTitle}>Filters</Text>
            <View style={styles.filterRow}>
              <Pressable
                onPress={() => setShowMistakesOnly((prev) => !prev)}
                style={[
                  styles.toggleChip,
                  showMistakesOnly ? styles.toggleChipActive : null,
                ]}
              >
                <Text
                  style={[
                    styles.toggleChipText,
                    showMistakesOnly ? styles.toggleChipTextActive : null,
                  ]}
                >
                  Only sessions with mistakes
                </Text>
              </Pressable>
            </View>
            <View style={styles.filterGrid}>
              <View style={styles.filterField}>
                <Text style={styles.filterLabel}>From</Text>
                <TextInput
                  value={dateFrom}
                  onChangeText={setDateFrom}
                  placeholder="YYYY-MM-DD"
                  placeholderTextColor={colors.textSoft}
                  autoCapitalize="none"
                  autoCorrect={false}
                  style={styles.dateInput}
                />
              </View>
              <View style={styles.filterField}>
                <Text style={styles.filterLabel}>To</Text>
                <TextInput
                  value={dateTo}
                  onChangeText={setDateTo}
                  placeholder="YYYY-MM-DD"
                  placeholderTextColor={colors.textSoft}
                  autoCapitalize="none"
                  autoCorrect={false}
                  style={styles.dateInput}
                />
              </View>
            </View>
            <View style={styles.filterGrid}>
              <SelectField
                label="Sort by"
                value={sortMode}
                options={sortOptions}
                onChange={(value) => setSortMode(value as SortMode)}
              />
            </View>
            {!dateFrom && !dateTo ? (
              <Text style={styles.filterHint}>Showing last 10 sessions by default.</Text>
            ) : null}
          </SectionCard>

          {sessions.length === 0 ? (
            <SectionCard>
              <Text style={styles.muted}>
                No sessions yet. Record or upload audio from the home screen first.
              </Text>
            </SectionCard>
          ) : filteredSessions.length === 0 ? (
            <SectionCard>
              <Text style={styles.muted}>No sessions match the current filters.</Text>
            </SectionCard>
          ) : sortMode === "mistake" ? (
            groupedSessions.map(([label, group]) => {
              const isOpen = openGroups.includes(label);
              return (
                <SectionCard key={label}>
                  <Pressable
                    onPress={() => toggleGroup(label)}
                    style={styles.groupHeader}
                  >
                    <Text style={styles.groupTitle}>
                      {label} ({group.length})
                    </Text>
                    <Text style={styles.groupToggle}>{isOpen ? "Hide" : "Show"}</Text>
                  </Pressable>
                  {isOpen ? group.map(renderSessionRow) : null}
                </SectionCard>
              );
            })
          ) : (
            <View>{filteredSessions.map(renderSessionRow)}</View>
          )}
        </>
      )}
    </Screen>
  );
}

export default memo(HistoryScreen);

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
  focusText: {
    color: colors.textMuted,
    marginTop: 6,
    fontSize: 13,
  },
  badge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 999,
  },
  badgeError: {
    backgroundColor: colors.redTint,
  },
  badgeMistakes: {
    backgroundColor: colors.yellowTint,
  },
  badgeClear: {
    backgroundColor: colors.greenTint,
  },
  badgeText: {
    fontWeight: "700",
    fontSize: 12,
  },
  badgeTextError: {
    color: "#991b1b",
  },
  badgeTextMistakes: {
    color: "#92400e",
  },
  badgeTextClear: {
    color: "#166534",
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
  filterRow: {
    marginBottom: 12,
  },
  toggleChip: {
    alignSelf: "flex-start",
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 999,
    backgroundColor: colors.surfaceMuted,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  toggleChipActive: {
    backgroundColor: colors.blueTint,
    borderColor: "#93c5fd",
  },
  toggleChipText: {
    color: colors.textMuted,
    fontSize: 13,
    fontWeight: "700",
  },
  toggleChipTextActive: {
    color: colors.primaryDark,
  },
  filterGrid: {
    flexDirection: "row",
    gap: 12,
    flexWrap: "wrap",
    marginBottom: 12,
  },
  filterField: {
    flex: 1,
    minWidth: 140,
  },
  filterLabel: {
    color: colors.textSoft,
    fontSize: 12,
    fontWeight: "700",
    marginBottom: 6,
  },
  dateInput: {
    minHeight: 44,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 12,
    backgroundColor: colors.surface,
    paddingHorizontal: 12,
    color: colors.text,
    fontSize: 14,
  },
  filterHint: {
    color: colors.textSoft,
    fontSize: 12,
  },
  groupHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8,
  },
  groupTitle: {
    color: colors.text,
    fontWeight: "800",
    fontSize: 16,
    flex: 1,
  },
  groupToggle: {
    color: colors.textSoft,
    fontWeight: "700",
  },
});
