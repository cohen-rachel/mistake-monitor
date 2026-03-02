import { useRef, useState, useCallback } from "react";
import { createTranscribeSocket } from "../services/api";
import type { TranscriptChunk } from "../types";

interface Props {
  onChunk: (text: string, fullText?: string) => void;
  onStatusChange: (recording: boolean) => void;
  language: string;
}

const btnBase: React.CSSProperties = {
  padding: "10px 20px",
  borderRadius: 8,
  border: "none",
  fontWeight: 600,
  fontSize: 15,
  cursor: "pointer",
  transition: "background 0.15s",
};

export default function AudioRecorder({ onChunk, onStatusChange, language }: Props) {
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Open WebSocket
      const ws = createTranscribeSocket(language);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const msg: TranscriptChunk = JSON.parse(event.data);
          if (msg.type === "transcript" && msg.text) {
            onChunk(msg.text, msg.full_text);
          }
        } catch {
          // ignore parse errors
        }
      };

      ws.onopen = () => {
        // Start MediaRecorder after WebSocket is ready
        const mediaRecorder = new MediaRecorder(stream, {
          mimeType: MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
            ? "audio/webm;codecs=opus"
            : "audio/webm",
        });
        mediaRecorderRef.current = mediaRecorder;

        mediaRecorder.ondataavailable = (event) => {
          if (
            event.data.size > 0 &&
            ws.readyState === WebSocket.OPEN
          ) {
            ws.send(event.data);
          }
        };

        mediaRecorder.start(3000); // 3-second chunks
        setIsRecording(true);
        onStatusChange(true);
      };

      ws.onerror = () => {
        stopRecording();
      };
    } catch (err) {
      console.error("Microphone access denied or unavailable:", err);
      alert("Could not access microphone. Please allow microphone access.");
    }
  }, [language, onChunk, onStatusChange]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
    }
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send("stop");
      wsRef.current.close();
    }
    mediaRecorderRef.current = null;
    wsRef.current = null;
    setIsRecording(false);
    onStatusChange(false);
  }, [onStatusChange]);

  return (
    <button
      onClick={isRecording ? stopRecording : startRecording}
      style={{
        ...btnBase,
        background: isRecording ? "#dc2626" : "#4338ca",
        color: "#fff",
      }}
    >
      {isRecording ? "⏹ Stop Recording" : "🎙 Start Recording"}
    </button>
  );
}
