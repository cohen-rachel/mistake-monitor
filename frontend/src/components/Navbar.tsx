import { NavLink } from "react-router-dom";
import { useState, useEffect } from "react";
import { UserLanguageProfileOut } from "../types";
import { api } from "../services/api";
import { useLanguageContext } from "../contexts/LanguageContext";

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
  const { currentLanguageProfile, setCurrentLanguageProfile, isLoadingLanguage } = useLanguageContext();
  const [languageProfiles, setLanguageProfiles] = useState<UserLanguageProfileOut[]>([]);

  useEffect(() => {
    const fetchLanguageProfiles = async () => {
      try {
        const profiles = await api.getUserLanguageProfiles();
        setLanguageProfiles(profiles);
        if (profiles.length > 0 && !currentLanguageProfile) {
          const current = await api.getCurrentLanguageProfile();
          if (current) {
            setCurrentLanguageProfile(current);
          } else {
            setCurrentLanguageProfile(profiles[0]);
          }
        }
      } catch (error) {
        console.error("Error fetching language profiles:", error);
      }
    };
    fetchLanguageProfiles();
  }, [currentLanguageProfile, setCurrentLanguageProfile]);

  const handleLanguageChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const profileId = parseInt(e.target.value);
    const selectedProfile = languageProfiles.find((p) => p.id === profileId);
    if (selectedProfile) {
      try {
        await api.setCurrentLanguageProfile(profileId);
        setCurrentLanguageProfile(selectedProfile);
      } catch (error) {
        console.error("Error setting current language profile:", error);
      }
    }
  };

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
      <NavLink to="/rewrite" style={linkStyle}>
        Rewrite
      </NavLink>
      <div style={{ marginLeft: "auto" }}>
      {isLoadingLanguage ? (
    <span style={{ color: "#ccc" }}>Loading Languages...</span>
  ) : languageProfiles.length > 0 && currentLanguageProfile ? (
    <select
      onChange={handleLanguageChange}
      value={currentLanguageProfile.id}
      style={{
        padding: "8px 12px",
        borderRadius: 6,
        border: "1px solid #333",
        background: "#2b2b40",
        color: "#fff",
        fontSize: 15,
      }}
    >
      {languageProfiles.map((profile) => (
        <option key={profile.id} value={profile.id}>
          {profile.display_name} ({profile.language_code.toUpperCase()})
        </option>
      ))}
    </select>
  ) : (
    <span style={{ color: "#ccc" }}>No language profiles yet</span>
  )}

        {/* {languageProfiles.length > 0 && currentLanguageProfile ? (
          <select
            onChange={handleLanguageChange}
            value={currentLanguageProfile.id}
            style={{
              padding: "8px 12px",
              borderRadius: 6,
              border: "1px solid #333",
              background: "#2b2b40",
              color: "#fff",
              fontSize: 15,
            }}
          >
            {languageProfiles.map((profile) => (
              <option key={profile.id} value={profile.id}>
                {profile.display_name} ({profile.language_code.toUpperCase()})
              </option>
            ))}
          </select>
        ) : (
          <span style={{ color: "#ccc" }}>Loading Languages...</span>
        )} */}
      </div>
    </nav>
  );
}
