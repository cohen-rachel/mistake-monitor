import { NavLink } from "react-router-dom";

const navStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 24,
  padding: "12px 24px",
  background: "#1a1a2e",
  color: "#fff",
  fontSize: 15,
  fontWeight: 500,
};

const linkBase: React.CSSProperties = {
  padding: "6px 12px",
  borderRadius: 6,
  transition: "background 0.15s",
};

export default function Navbar() {
  const linkStyle = ({ isActive }: { isActive: boolean }): React.CSSProperties => ({
    ...linkBase,
    background: isActive ? "rgba(255,255,255,0.15)" : "transparent",
  });

  return (
    <nav style={navStyle}>
      <span style={{ fontWeight: 700, fontSize: 18, marginRight: 16 }}>
        Language Tutor
      </span>
      <NavLink to="/" style={linkStyle} end>
        Home
      </NavLink>
      <NavLink to="/history" style={linkStyle}>
        History
      </NavLink>
      <NavLink to="/insights" style={linkStyle}>
        Insights
      </NavLink>
      <NavLink to="/practice" style={linkStyle}>
        Practice
      </NavLink>
    </nav>
  );
}
