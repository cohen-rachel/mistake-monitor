export default function Practice() {
  return (
    <div>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 16 }}>
        Practice
      </h1>

      <div
        style={{
          background: "#fff",
          border: "1px solid #e2e8f0",
          borderRadius: 12,
          padding: 40,
          textAlign: "center",
        }}
      >
        <div style={{ fontSize: 48, marginBottom: 16 }}>📝</div>
        <h2 style={{ fontSize: 20, fontWeight: 600, marginBottom: 8 }}>
          Coming Soon
        </h2>
        <p
          style={{
            color: "#64748b",
            maxWidth: 480,
            margin: "0 auto",
            lineHeight: 1.7,
          }}
        >
          This page will generate personalized practice exercises based on your
          most common mistakes. It will analyze your error patterns and create
          targeted fill-in-the-blank, correction, and rephrasing exercises to
          help you improve.
        </p>
      </div>
    </div>
  );
}
