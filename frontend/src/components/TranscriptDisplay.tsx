interface Props {
  transcript: string;
  isRecording: boolean;
  onTranscriptChange?: (updated: string) => void;
}

const containerStyle: React.CSSProperties = {
  background: "#fff",
  border: "1px solid #e2e8f0",
  borderRadius: 8,
  padding: 16,
  minHeight: 120,
  maxHeight: 300,
  fontFamily: "'SF Mono', 'Fira Code', monospace",
  fontSize: 14,
  lineHeight: 1.8,
  color: "#334155",
  position: "relative",
};

const textareaStyle: React.CSSProperties = {
  width: "100%",
  height: "100%",
  minHeight: 120,
  maxHeight: 300,
  background: "transparent",
  border: "none",
  resize: "vertical",
  padding: 0,
  font: "inherit",
  outline: "none",
  color: "inherit",
};

const indicatorStyle: React.CSSProperties = {
  position: "absolute",
  top: 16,
  right: 16,
  width: 8,
  height: 16,
  background: "#4338ca",
  animation: "blink 1s infinite",
};

export default function TranscriptDisplay({
  transcript,
  isRecording,
  onTranscriptChange,
}: Props) {
  return (
    <div style={containerStyle}>
      <textarea
        value={transcript}
        onChange={(event) => onTranscriptChange?.(event.target.value)}
        placeholder="Transcript will appear here..."
        disabled={isRecording}
        style={textareaStyle}
      />
      {isRecording && <span style={indicatorStyle} />}
      <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }`}</style>
    </div>
  );
}
