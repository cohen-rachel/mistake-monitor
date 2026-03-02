import { useState, useCallback, useRef } from "react";
import AudioRecorder from "../components/AudioRecorder";
import TranscriptDisplay from "../components/TranscriptDisplay";
import MistakeCard from "../components/MistakeCard";
import {
  createSessionWithTranscript,
  createSessionWithAudio,
  analyzeSession,
} from "../services/api";
import type { MistakeOut } from "../types";

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

export default function Landing() {
  const [tab, setTab] = useState<"record" | "upload">("record");
  const [language, setLanguage] = useState("en");

  // -- Record tab state --
  const [isRecording, setIsRecording] = useState(false);
  const [liveTranscript, setLiveTranscript] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [mistakes, setMistakes] = useState<MistakeOut[]>([]);
  const [statusMsg, setStatusMsg] = useState("");

  // -- Upload tab state --
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadAnalyzing, setUploadAnalyzing] = useState(false);
  const [uploadMistakes, setUploadMistakes] = useState<MistakeOut[]>([]);
  const [uploadTranscript, setUploadTranscript] = useState("");
  const [uploadStatus, setUploadStatus] = useState("");

  const fileInputRef = useRef<HTMLInputElement>(null);

  // -- Record handlers --
  const handleChunk = useCallback((text: string, fullText?: string) => {
    if (fullText) {
      setLiveTranscript(fullText);
      return;
    }
    setLiveTranscript((prev) => `${prev} ${text}`.trim());
  }, []);

  const handleStatusChange = useCallback((recording: boolean) => {
    setIsRecording(recording);
  }, []);

  const handleAnalyzeRecording = async () => {
    const fullText = liveTranscript.trim();
    if (!fullText) {
      setStatusMsg("No transcript to analyze. Record something first.");
      return;
    }

    setAnalyzing(true);
    setStatusMsg("Saving session...");
    try {
      const session = await createSessionWithTranscript(fullText, language);
      setStatusMsg("Analyzing...");
      const result = await analyzeSession(session.id);
      setMistakes(result.mistakes);
      setStatusMsg(
        result.mistakes.length > 0
          ? `Found ${result.mistakes.length} mistake(s).`
          : "No mistakes found!"
      );
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
  };

  // -- Upload handlers --
  const handleUploadAnalyze = async () => {
    if (!uploadFile) return;
    setUploadAnalyzing(true);
    setUploadStatus("Uploading and analyzing...");
    try {
      const session = await createSessionWithAudio(uploadFile, language);
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
        Record yourself speaking or upload an audio file to get feedback on
        grammar and vocabulary.
      </p>

      {/* Tabs */}
      <div style={tabBarStyle}>
        <button
          style={{ ...tabStyle(tab === "record"), borderRadius: "8px 0 0 8px" }}
          onClick={() => setTab("record")}
        >
          Real-time Recording
        </button>
        <button
          style={{ ...tabStyle(tab === "upload"), borderRadius: "0 8px 8px 0" }}
          onClick={() => setTab("upload")}
        >
          Upload Audio File
        </button>
      </div>

      <div style={{ marginBottom: 16 }}>
        <label style={{ fontSize: 14, color: "#475569", marginRight: 8 }}>
          Spoken language:
        </label>
        <select
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          style={{
            padding: "8px 10px",
            borderRadius: 8,
            border: "1px solid #cbd5e1",
            background: "#fff",
            color: "#1e293b",
          }}
        >
          <option value="en">English</option>
          <option value="es">Spanish</option>
          <option value="fr">French</option>
          <option value="de">German</option>
          <option value="it">Italian</option>
          <option value="pt">Portuguese</option>
          <option value="nl">Dutch</option>
          <option value="ru">Russian</option>
          <option value="uk">Ukrainian</option>
          <option value="pl">Polish</option>
          <option value="tr">Turkish</option>
          <option value="ar">Arabic</option>
          <option value="he">Hebrew</option>
          <option value="hi">Hindi</option>
          <option value="ja">Japanese</option>
          <option value="ko">Korean</option>
          <option value="zh">Chinese</option>
        </select>
      </div>

      {/* ===== RECORD TAB ===== */}
      {tab === "record" && (
        <div>
          <div style={sectionStyle}>
            <div
              style={{
                display: "flex",
                gap: 12,
                alignItems: "center",
                marginBottom: 16,
              }}
            >
              <AudioRecorder
                onChunk={handleChunk}
                onStatusChange={handleStatusChange}
                language={language}
              />
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

      {/* ===== UPLOAD TAB ===== */}
      {tab === "upload" && (
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
              style={
                !uploadFile || uploadAnalyzing ? btnDisabled : btnPrimary
              }
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
