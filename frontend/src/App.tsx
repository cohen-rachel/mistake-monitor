import { Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Landing from "./pages/Landing";
import History from "./pages/History";
import Insights from "./pages/Insights";

const globalStyles = `
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,
      Ubuntu, Cantarell, sans-serif;
    background: #f5f7fa;
    color: #1a1a2e;
    line-height: 1.6;
  }
  a { color: inherit; text-decoration: none; }
`;

export default function App() {
  return (
    <>
      <style>{globalStyles}</style>
      <Navbar />
      <main style={{ maxWidth: 960, margin: "0 auto", padding: "24px 16px" }}>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/history" element={<History />} />
          <Route path="/insights" element={<Insights />} />
        </Routes>
      </main>
    </>
  );
}
