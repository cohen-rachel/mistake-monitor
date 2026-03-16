import React, {
  createContext,
  ReactNode,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { UserLanguageProfileOut } from "../types";
import { api } from "../services/api";

interface LanguageContextType {
  currentLanguageProfile: UserLanguageProfileOut | null;
  setCurrentLanguageProfile: (profile: UserLanguageProfileOut | null) => void;
  isLoadingLanguage: boolean;
  languageProfiles: UserLanguageProfileOut[];
  refreshLanguageProfiles: () => Promise<void>;
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [currentLanguageProfile, setCurrentLanguageProfile] =
    useState<UserLanguageProfileOut | null>(null);
  const [languageProfiles, setLanguageProfiles] = useState<UserLanguageProfileOut[]>([]);
  const [isLoadingLanguage, setIsLoadingLanguage] = useState(true);

  const refreshLanguageProfiles = async () => {
    const profiles = await api.getUserLanguageProfiles();
    setLanguageProfiles(profiles);
    if (profiles.length === 0) {
      setCurrentLanguageProfile(null);
      return;
    }
    const current = await api.getCurrentLanguageProfile();
    if (current) {
      setCurrentLanguageProfile(current);
      return;
    }
    setCurrentLanguageProfile(profiles[0]);
    await api.setCurrentLanguageProfile(profiles[0].id);
  };

  useEffect(() => {
    let mounted = true;
    void (async () => {
      try {
        await refreshLanguageProfiles();
      } catch (error) {
        if (__DEV__) {
          console.warn("Failed to load language profiles", error);
        }
        if (mounted) {
          setLanguageProfiles([]);
          setCurrentLanguageProfile(null);
        }
      } finally {
        if (mounted) {
          setIsLoadingLanguage(false);
        }
      }
    })();

    return () => {
      mounted = false;
    };
  }, []);

  const value = useMemo(
    () => ({
      currentLanguageProfile,
      setCurrentLanguageProfile,
      isLoadingLanguage,
      languageProfiles,
      refreshLanguageProfiles,
    }),
    [
      currentLanguageProfile,
      isLoadingLanguage,
      languageProfiles,
    ]
  );

  return (
    <LanguageContext.Provider value={value}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguageContext() {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error("useLanguageContext must be used within LanguageProvider");
  }
  return context;
}
