import React, { createContext, useContext, useState, ReactNode, useEffect } from "react";
import { UserLanguageProfileOut } from "../types";
import { api } from "../services/api";

interface LanguageContextType {
  currentLanguageProfile: UserLanguageProfileOut | null;
  setCurrentLanguageProfile: (profile: UserLanguageProfileOut | null) => void;
  isLoadingLanguage: boolean;
  
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export const LanguageProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [currentLanguageProfile, setCurrentLanguageProfile] = useState<UserLanguageProfileOut | null>(null);
  const [isLoadingLanguage, setIsLoadingLanguage] = useState<boolean>(true);

  useEffect(() => {
    const fetchInitialLanguageProfile = async () => {
      try {
        const profiles = await api.getUserLanguageProfiles();
        if (profiles.length > 0) {
          const current = await api.getCurrentLanguageProfile();
          if (current) {
            setCurrentLanguageProfile(current);
          } else {
            // If no current is set, default to the first available
            setCurrentLanguageProfile(profiles[0]);
            await api.setCurrentLanguageProfile(profiles[0].id);
          }
        }
      } catch (error) {
        console.error("Error fetching initial language profile:", error);
      } finally {
        setIsLoadingLanguage(false);
      }
    };
    fetchInitialLanguageProfile();
  }, []);

  return (
    <LanguageContext.Provider
      value={{
        currentLanguageProfile,
        setCurrentLanguageProfile,
        isLoadingLanguage,
      }}
    >
      {children}
    </LanguageContext.Provider>
  );
};

export const useLanguageContext = () => {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error("useLanguageContext must be used within a LanguageProvider");
  }
  return context;
};
