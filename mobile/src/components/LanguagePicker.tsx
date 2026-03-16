import React, { useMemo } from "react";
import { StyleSheet, Text, View } from "react-native";
import { useLanguageContext } from "../contexts/LanguageContext";
import { api } from "../services/api";
import SelectField from "./SelectField";
import { colors } from "../theme";

export default function LanguagePicker() {
  const {
    currentLanguageProfile,
    isLoadingLanguage,
    languageProfiles,
    setCurrentLanguageProfile,
  } = useLanguageContext();

  const options = useMemo(
    () =>
      languageProfiles.map((profile) => ({
        label: profile.display_name,
        value: String(profile.id),
        subtitle: profile.language_code.toUpperCase(),
      })),
    [languageProfiles]
  );

  const handleLanguageChange = async (value: string) => {
    const profileId = Number(value);
    const selectedProfile = languageProfiles.find((item) => item.id === profileId);
    if (!selectedProfile) {
      return;
    }
    await api.setCurrentLanguageProfile(selectedProfile.id);
    setCurrentLanguageProfile(selectedProfile);
  };

  if (isLoadingLanguage) {
    return <Text style={styles.label}>Loading languages...</Text>;
  }

  if (!currentLanguageProfile) {
    return <Text style={styles.label}>No language profiles</Text>;
  }

  return (
    <View style={styles.wrapper}>
      <SelectField
        label="Profile"
        value={String(currentLanguageProfile.id)}
        options={options}
        onChange={(value) => {
          void handleLanguageChange(value);
        }}
        disabled={languageProfiles.length === 0}
        variant="dark"
        compact
      />
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: {
    width: 124,
  },
  label: {
    color: colors.white,
    fontSize: 12,
    fontWeight: "600",
  },
});
