import { useRef, useState, useCallback } from "react";
import {
  createTranscribeSocket,
  finalizeRecordedAudio,
  getTranscriptionConfig,
} from "../services/api";
import type { TranscriptChunk } from "../types";

interface Props {
  onChunk: (text: string, fullText?: string) => void;
  onStatusChange: (recording: boolean) => void;
  language?: string;
  onFinalTranscript?: (analysisText: string, displayText: string) => void;
  onError?: (message: string) => void;
  onFinalizeStateChange?: (finalizing: boolean) => void;
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

export default function AudioRecorder({
  onChunk,
  onStatusChange,
  language,
  onFinalTranscript,
  onError,
  onFinalizeStateChange,
}: Props) {
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const mimeTypeRef = useRef<string>("audio/webm");
  const skipFinalPassRef = useRef(false);

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

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      chunksRef.current = [];
      try {
        const config = await getTranscriptionConfig();
        skipFinalPassRef.current = config.skip_final_pass;
      } catch {
        skipFinalPassRef.current = false;
      }

      const ws = createTranscribeSocket();
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

      ws.onerror = () => {
        onError?.("Live transcription is unavailable. Recording will continue and the final transcript will appear after you stop.");
        try {
          ws.close();
        } catch {
          // ignore close errors
        }
      };

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
          ? "audio/webm;codecs=opus"
          : "audio/webm",
      });
      mimeTypeRef.current = mediaRecorder.mimeType || "audio/webm";
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(event.data);
          }
        }
      };

      mediaRecorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, {
          type: mimeTypeRef.current || "audio/webm",
        });
        chunksRef.current = [];
        if (skipFinalPassRef.current) {
          onFinalTranscript?.("", "");
          onFinalizeStateChange?.(false);
          return;
        }
        if (!blob.size) {
          onFinalizeStateChange?.(false);
          return;
        }
        onFinalizeStateChange?.(true);
        try {
          const extension = mimeTypeRef.current.includes("mp4") ? "m4a" : "webm";
          const file = new File([blob], `recording.${extension}`, {
            type: mimeTypeRef.current || "audio/webm",
          });
          const result = await finalizeRecordedAudio(file, language);
          onFinalTranscript?.(result.analysis_text, result.display_text);
        } catch (err) {
          const message =
            err instanceof Error ? err.message : "Final transcription failed.";
          onError?.(message);
        } finally {
          onFinalizeStateChange?.(false);
        }
      };

      mediaRecorder.start(3000); // 3-second chunks
      setIsRecording(true);
      onStatusChange(true);
    } catch (err) {
      console.error("Microphone access denied or unavailable:", err);
      alert("Could not access microphone. Please allow microphone access.");
    }
  }, [language, onChunk, onError, onFinalTranscript, onFinalizeStateChange, onStatusChange]);

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
