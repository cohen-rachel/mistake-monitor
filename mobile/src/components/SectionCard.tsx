import React, { ReactNode } from "react";
import { StyleSheet, View } from "react-native";
import { colors, layout } from "../theme";

export default function SectionCard({ children }: { children: ReactNode }) {
  return <View style={styles.card}>{children}</View>;
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.surface,
    borderRadius: layout.radius,
    borderWidth: 1,
    borderColor: colors.border,
    padding: 16,
    marginBottom: layout.sectionGap,
  },
});
