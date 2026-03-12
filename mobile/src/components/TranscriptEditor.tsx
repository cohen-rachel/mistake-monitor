import React from "react";
import { StyleSheet, TextInput, View } from "react-native";
import { colors } from "../theme";

interface Props {
  transcript: string;
  editable?: boolean;
  onChangeText?: (value: string) => void;
}

export default function TranscriptEditor({
  transcript,
  editable = true,
  onChangeText,
}: Props) {
  return (
    <View style={styles.container}>
      <TextInput
        style={styles.input}
        multiline
        editable={editable}
        onChangeText={onChangeText}
        value={transcript}
        placeholder="Transcript will appear here, or type your own text to analyze."
        placeholderTextColor={colors.textSoft}
        textAlignVertical="top"
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 12,
    minHeight: 140,
    padding: 12,
  },
  input: {
    minHeight: 116,
    color: colors.text,
    fontSize: 15,
    lineHeight: 22,
  },
});
