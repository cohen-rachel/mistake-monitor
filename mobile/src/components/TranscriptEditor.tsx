import React, { useState } from "react";
import { Keyboard, Pressable, StyleSheet, Text, TextInput, View } from "react-native";
import { colors } from "../theme";

interface Props {
  transcript: string;
  editable?: boolean;
  onChangeText?: (value: string) => void;
  onFocus?: () => void;
}

export default function TranscriptEditor({
  transcript,
  editable = true,
  onChangeText,
  onFocus,
}: Props) {
  const [isFocused, setIsFocused] = useState(false);

  return (
    <View style={styles.wrapper}>
      <View style={[styles.container, isFocused && styles.containerFocused]}>
        <TextInput
          style={styles.input}
          multiline
          editable={editable}
          onChangeText={onChangeText}
          value={transcript}
          placeholder="Transcript will appear here, or type your own text to analyze."
          placeholderTextColor={colors.textSoft}
          textAlignVertical="top"
          scrollEnabled
          onFocus={() => {
            setIsFocused(true);
            onFocus?.();
          }}
          onBlur={() => setIsFocused(false)}
        />
      </View>
      {editable && isFocused ? (
        <Pressable
          style={styles.dismissButton}
          onPress={() => {
            Keyboard.dismiss();
            setIsFocused(false);
          }}
        >
          <Text style={styles.dismissText}>Done</Text>
        </Pressable>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: {
    gap: 10,
  },
  container: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 12,
    minHeight: 140,
    padding: 12,
  },
  containerFocused: {
    borderColor: colors.primary,
  },
  input: {
    minHeight: 116,
    maxHeight: 220,
    color: colors.text,
    fontSize: 15,
    lineHeight: 22,
  },
  dismissButton: {
    alignSelf: "flex-end",
    backgroundColor: colors.primary,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 999,
  },
  dismissText: {
    color: colors.white,
    fontSize: 13,
    fontWeight: "700",
  },
});
