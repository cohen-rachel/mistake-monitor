import { useEffect, useState } from "react";
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

const statusBadge = (status: string): React.CSSProperties => ({
  display: "inline-block",
  padding: "2px 8px",
  borderRadius: 12,
  fontSize: 12,
  fontWeight: 600,
  background:
    status === "analyzed"
      ? "#dcfce7"
      : status === "transcribed"
      ? "#e0f2fe"
      : "#f1f5f9",
  color:
    status === "analyzed"
      ? "#166534"
      : status === "transcribed"
      ? "#0369a1"
      : "#475569",
});

export default function History() {
  const { currentLanguageProfile, isLoadingLanguage } = useLanguageContext();
  const [sessions, setSessions] = useState<SessionOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [detail, setDetail] = useState<SessionDetailOut | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

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
  const profile = currentLanguageProfile;
  return (
    <div>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 16 }}>
        Session History ({currentLanguageProfile.display_name})
      </h1>

      {sessions.length === 0 ? (
        <p style={{ color: "#94a3b8" }}>
          No sessions yet. Go to the home page to record or upload audio.
        </p>
      ) : (
        sessions.map((s) => (
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
                }}
              >
                <div>
                  <span style={{ fontWeight: 600 }}>Session #{s.id}</span>
                  <span
                    style={{
                      marginLeft: 12,
                      fontSize: 13,
                      color: "#64748b",
                    }}
                  >
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
                </div>
                <span style={statusBadge(s.status)}>{s.status}</span>
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
        ))
      )}
    </div>
  );
}
