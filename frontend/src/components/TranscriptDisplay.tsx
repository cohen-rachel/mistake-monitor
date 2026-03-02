interface Props {
  transcript: string;
  isRecording: boolean;
}

const containerStyle: React.CSSProperties = {
  background: "#fff",
  border: "1px solid #e2e8f0",
  borderRadius: 8,
  padding: 16,
  minHeight: 120,
  maxHeight: 300,
  overflowY: "auto",
  fontFamily: "'SF Mono', 'Fira Code', monospace",
  fontSize: 14,
  lineHeight: 1.8,
  color: "#334155",
};

export default function TranscriptDisplay({ transcript, isRecording }: Props) {
  if (!transcript && !isRecording) {
    return (
      <div style={{ ...containerStyle, color: "#94a3b8" }}>
        Transcript will appear here...
      </div>
    );
  }

  return (
    <div style={containerStyle}>
      <span>{transcript} </span>
      {isRecording && (
        <span
          style={{
            display: "inline-block",
            width: 8,
            height: 16,
            background: "#4338ca",
            animation: "blink 1s infinite",
            verticalAlign: "text-bottom",
            marginLeft: 2,
          }}
        />
      )}
      <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }`}</style>
    </div>
  );
}
