import React, { useMemo, useState } from "react";
import {
  SafeAreaView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { StatusBar as ExpoStatusBar } from "expo-status-bar";
import LandingScreen from "./src/screens/LandingScreen";
import HistoryScreen from "./src/screens/HistoryScreen";
import InsightsScreen from "./src/screens/InsightsScreen";
import RewriteScreen from "./src/screens/RewriteScreen";
import { LanguageProvider, useLanguageContext } from "./src/contexts/LanguageContext";
import { LandingStateProvider } from "./src/contexts/LandingStateContext";
import LanguagePicker from "./src/components/LanguagePicker";
import { colors, layout } from "./src/theme";

type ScreenKey = "home" | "history" | "insights" | "rewrite";

const tabs: Array<{ key: ScreenKey; label: string }> = [
  { key: "home", label: "Home" },
  { key: "history", label: "History" },
  { key: "insights", label: "Insights" },
  { key: "rewrite", label: "Rewrite" },
];

function AppShell() {
  const [screen, setScreen] = useState<ScreenKey>("home");
  const [visited, setVisited] = useState<Record<ScreenKey, boolean>>({
    home: true,
    history: false,
    insights: false,
    rewrite: false,
  });
  const { currentLanguageProfile } = useLanguageContext();

  const renderedScreens = useMemo(
    () => ({
      home: <LandingScreen />,
      history: <HistoryScreen />,
      insights: <InsightsScreen />,
      rewrite: <RewriteScreen />,
    }),
    []
  );

  return (
    <SafeAreaView style={styles.safeArea}>
      <ExpoStatusBar style="light" />
      <StatusBar barStyle="light-content" />
      <View style={styles.header}>
        <View style={styles.headerTop}>
          <View>
            <Text style={styles.brand}>Language Tutor</Text>
            <Text style={styles.subtitle}>
              {currentLanguageProfile
                ? currentLanguageProfile.display_name
                : "Choose a language profile"}
            </Text>
          </View>
          <LanguagePicker />
        </View>
        <View style={styles.tabRow}>
          {tabs.map((tab) => {
            const active = tab.key === screen;
            return (
              <TouchableOpacity
                key={tab.key}
                onPress={() => {
                  setVisited((prev) =>
                    prev[tab.key] ? prev : { ...prev, [tab.key]: true }
                  );
                  setScreen(tab.key);
                }}
                style={[styles.tab, active && styles.tabActive]}
              >
                <Text style={[styles.tabText, active && styles.tabTextActive]}>
                  {tab.label}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>
      </View>
      <LandingStateProvider>
        <View style={styles.screenContainer}>
          {tabs.map((tab) => {
            if (!visited[tab.key]) {
              return null;
            }
            const active = tab.key === screen;
            return (
              <View
                key={tab.key}
                style={[styles.screenPane, !active && styles.screenPaneHidden]}
                pointerEvents={active ? "auto" : "none"}
              >
                {renderedScreens[tab.key]}
              </View>
            );
          })}
        </View>
      </LandingStateProvider>
    </SafeAreaView>
  );
}

export default function App() {
  return (
    <LanguageProvider>
      <AppShell />
    </LanguageProvider>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: colors.ink,
  },
  header: {
    backgroundColor: colors.ink,
    paddingHorizontal: layout.screenPadding,
    paddingTop: 8,
    paddingBottom: 14,
  },
  headerTop: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    gap: 12,
    marginBottom: 14,
  },
  brand: {
    color: colors.white,
    fontSize: 26,
    fontWeight: "800",
  },
  subtitle: {
    color: colors.inkMuted,
    fontSize: 13,
    marginTop: 4,
  },
  tabRow: {
    flexDirection: "row",
    gap: 8,
    flexWrap: "wrap",
  },
  tab: {
    paddingHorizontal: 14,
    paddingVertical: 9,
    borderRadius: 999,
    backgroundColor: colors.inkSoft,
  },
  tabActive: {
    backgroundColor: colors.primary,
  },
  tabText: {
    color: colors.inkMuted,
    fontSize: 13,
    fontWeight: "700",
  },
  tabTextActive: {
    color: colors.white,
  },
  screenContainer: {
    flex: 1,
  },
  screenPane: {
    flex: 1,
  },
  screenPaneHidden: {
    display: "none",
  },
});
