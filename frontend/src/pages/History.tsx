import { useEffect, useMemo, useState } from "react";
import { listSessions, getSession } from "../services/api";
import MistakeCard from "../components/MistakeCard";
import type { SessionOut, SessionDetailOut } from "../types";
import { useLanguageContext } from "../contexts/LanguageContext";
const rowStyle: React.CSSProperties = {
  background: "#fff",
  border: "1px solid #e2e8f0",
  borderRadius: 8,
  padding: 16,
  marginBottom: 8,
  cursor: "pointer",
  transition: "box-shadow 0.15s",
};

const statusBadge = (tone: "error" | "mistakes" | "clear"): React.CSSProperties => ({
  display: "inline-block",
  padding: "2px 8px",
  borderRadius: 12,
  fontSize: 12,
  fontWeight: 600,
  background:
    tone === "error" ? "#fee2e2" : tone === "mistakes" ? "#fef3c7" : "#dcfce7",
  color:
    tone === "error" ? "#991b1b" : tone === "mistakes" ? "#92400e" : "#166534",
});

const controlStyle: React.CSSProperties = {
  padding: "8px 10px",
  borderRadius: 8,
  border: "1px solid #cbd5e1",
  background: "#fff",
  color: "#334155",
  fontSize: 13,
};

function formatSessionBadge(session: SessionOut): { label: string; tone: "error" | "mistakes" | "clear" } {
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

export default function History() {
  const { currentLanguageProfile, isLoadingLanguage } = useLanguageContext();
  const [sessions, setSessions] = useState<SessionOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [detail, setDetail] = useState<SessionDetailOut | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [showMistakesOnly, setShowMistakesOnly] = useState(false);
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [sortMode, setSortMode] = useState<"date" | "mistake">("date");

  useEffect(() => {
    const profile = currentLanguageProfile;
    if (!profile) {
      if (!isLoadingLanguage) setLoading(false);
      return;
    }
    setLoading(true);
    listSessions(profile.id)
      .then((data) => setSessions(data.sessions))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [currentLanguageProfile, isLoadingLanguage]);

  const filteredSessions = useMemo(() => {
    let next = [...sessions];

    if (showMistakesOnly) {
      next = next.filter((session) => session.mistake_count > 0);
    }

    if (dateFrom) {
      const from = new Date(`${dateFrom}T00:00:00`);
      next = next.filter((session) => new Date(session.created_at) >= from);
    }

    if (dateTo) {
      const to = new Date(`${dateTo}T23:59:59`);
      next = next.filter((session) => new Date(session.created_at) <= to);
    }

    if (sortMode === "date") {
      next.sort(
        (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      );
      if (!dateFrom && !dateTo) {
        next = next.slice(0, 10);
      }
      return next;
    }

    next.sort((a, b) => {
      const aLabel = a.primary_focus_label || (a.mistake_count > 0 ? "Other mistakes" : "No mistakes");
      const bLabel = b.primary_focus_label || (b.mistake_count > 0 ? "Other mistakes" : "No mistakes");
      if (aLabel !== bLabel) {
        return aLabel.localeCompare(bLabel);
      }
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
    });
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
          ? session.primary_focus_label || "Other mistakes"
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
      const d = await getSession(id);
      setDetail(d);
    } catch (err) {
      console.error(err);
    } finally {
      setDetailLoading(false);
    }
  };

  if (loading || isLoadingLanguage) {
    return <p style={{ color: "#94a3b8", textAlign: "center" }}>Loading...</p>;
  }

  if (!currentLanguageProfile) {
    return (
      <p style={{ color: "#94a3b8", textAlign: "center" }}>
        Please select or create a language profile to view history.
      </p>
    );
  }
  const renderSessionRow = (s: SessionOut) => {
    const badge = formatSessionBadge(s);
    return (
      <div key={s.id}>
        <div
          style={rowStyle}
          onClick={() => handleExpand(s.id)}
          onMouseEnter={(e) =>
            (e.currentTarget.style.boxShadow =
              "0 2px 8px rgba(0,0,0,0.08)")
          }
          onMouseLeave={(e) =>
            (e.currentTarget.style.boxShadow = "none")
          }
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              gap: 16,
            }}
          >
            <div>
              <span style={{ fontWeight: 600 }}>
                {new Date(s.created_at).toLocaleString()}
              </span>
              <span
                style={{
                  marginLeft: 12,
                  fontSize: 13,
                  color: "#64748b",
                }}
              >
                Language: {s.language}
              </span>
              {s.primary_focus_label && s.mistake_count > 0 ? (
                <div style={{ fontSize: 13, color: "#475569", marginTop: 6 }}>
                  Focus: {s.primary_focus_label}
                </div>
              ) : null}
            </div>
            <span style={statusBadge(badge.tone)}>{badge.label}</span>
          </div>
        </div>

        {expandedId === s.id && (
          <div
            style={{
              background: "#f8fafc",
              border: "1px solid #e2e8f0",
              borderTop: "none",
              borderRadius: "0 0 8px 8px",
              padding: 16,
              marginBottom: 8,
              marginTop: -8,
            }}
          >
            {detailLoading ? (
              <p style={{ color: "#94a3b8" }}>Loading details...</p>
            ) : detail ? (
              <>
                {detail.transcript && (
                  <div style={{ marginBottom: 16 }}>
                    <h3
                      style={{
                        fontSize: 15,
                        fontWeight: 600,
                        marginBottom: 8,
                      }}
                    >
                      Transcript
                    </h3>
                    <p
                      style={{
                        fontSize: 14,
                        color: "#334155",
                        lineHeight: 1.8,
                        background: "#fff",
                        padding: 12,
                        borderRadius: 6,
                        border: "1px solid #e2e8f0",
                      }}
                    >
                      {detail.transcript.raw_text}
                    </p>
                  </div>
                )}

                {detail.mistakes.length > 0 ? (
                  <div>
                    <h3
                      style={{
                        fontSize: 15,
                        fontWeight: 600,
                        marginBottom: 8,
                      }}
                    >
                      Mistakes ({detail.mistakes.length})
                    </h3>
                    {detail.mistakes.map((m) => (
                      <MistakeCard key={m.id} mistake={m} />
                    ))}
                  </div>
                ) : (
                  <p style={{ color: "#94a3b8", fontSize: 14 }}>
                    No mistakes found for this session.
                  </p>
                )}
              </>
            ) : null}
          </div>
        )}
      </div>
    );
  };

  return (
    <div>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 16 }}>
        Session History ({currentLanguageProfile.display_name})
      </h1>

      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 12,
          alignItems: "center",
          marginBottom: 18,
          padding: 12,
          borderRadius: 8,
          border: "1px solid #e2e8f0",
          background: "#f8fafc",
        }}
      >
        <label style={{ display: "inline-flex", alignItems: "center", gap: 8, color: "#334155", fontSize: 13 }}>
          <input
            type="checkbox"
            checked={showMistakesOnly}
            onChange={(event) => setShowMistakesOnly(event.target.checked)}
          />
          Only sessions with mistakes
        </label>
        <label style={{ display: "inline-flex", alignItems: "center", gap: 8, color: "#334155", fontSize: 13 }}>
          From
          <input
            type="date"
            value={dateFrom}
            onChange={(event) => setDateFrom(event.target.value)}
            style={controlStyle}
          />
        </label>
        <label style={{ display: "inline-flex", alignItems: "center", gap: 8, color: "#334155", fontSize: 13 }}>
          To
          <input
            type="date"
            value={dateTo}
            onChange={(event) => setDateTo(event.target.value)}
            style={controlStyle}
          />
        </label>
        <label style={{ display: "inline-flex", alignItems: "center", gap: 8, color: "#334155", fontSize: 13 }}>
          Sort by
          <select
            value={sortMode}
            onChange={(event) => setSortMode(event.target.value as "date" | "mistake")}
            style={controlStyle}
          >
            <option value="date">Date</option>
            <option value="mistake">Mistake type</option>
          </select>
        </label>
        {!dateFrom && !dateTo ? (
          <span style={{ fontSize: 12, color: "#64748b" }}>Showing last 10 sessions by default.</span>
        ) : null}
      </div>

      {sessions.length === 0 ? (
        <p style={{ color: "#94a3b8" }}>
          No sessions yet. Go to the home page to record or upload audio.
        </p>
      ) : filteredSessions.length === 0 ? (
        <p style={{ color: "#94a3b8" }}>
          No sessions match the current filters.
        </p>
      ) : sortMode === "mistake" ? (
        groupedSessions.map(([label, group]) => (
          <details key={label} style={{ marginBottom: 12 }}>
            <summary style={{ cursor: "pointer", fontWeight: 600, color: "#334155" }}>
              {label} ({group.length})
            </summary>
            <div style={{ marginTop: 10 }}>
              {group.map(renderSessionRow)}
            </div>
          </details>
        ))
      ) : (
        filteredSessions.map(renderSessionRow)
      )}
    </div>
  );
}
