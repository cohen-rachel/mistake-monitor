import React, { useMemo, useState } from "react";
import {
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { colors } from "../theme";

export interface SelectOption {
  label: string;
  value: string;
  subtitle?: string;
}

interface Props {
  label?: string;
  value: string;
  options: SelectOption[];
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  variant?: "light" | "dark";
}

export default function SelectField({
  label,
  value,
  options,
  onChange,
  placeholder = "Select...",
  disabled = false,
  variant = "light",
}: Props) {
  const [open, setOpen] = useState(false);

  const selected = useMemo(
    () => options.find((option) => option.value === value) ?? null,
    [options, value]
  );

  const palette =
    variant === "dark"
      ? {
          border: "#334155",
          background: "#1e293b",
          text: colors.white,
          subtext: colors.inkMuted,
          modalBackground: "#0f172a",
          modalBorder: "#334155",
        }
      : {
          border: colors.border,
          background: colors.surface,
          text: colors.text,
          subtext: colors.textSoft,
          modalBackground: colors.surface,
          modalBorder: colors.border,
        };

  return (
    <View style={styles.container}>
      {label ? <Text style={[styles.label, { color: palette.subtext }]}>{label}</Text> : null}
      <TouchableOpacity
        disabled={disabled}
        onPress={() => setOpen(true)}
        style={[
          styles.trigger,
          {
            borderColor: palette.border,
            backgroundColor: disabled ? colors.surfaceMuted : palette.background,
          },
        ]}
      >
        <View style={styles.triggerCopy}>
          <Text style={[styles.triggerText, { color: palette.text }]} numberOfLines={1}>
            {selected?.label || placeholder}
          </Text>
          {selected?.subtitle ? (
            <Text style={[styles.triggerSubtitle, { color: palette.subtext }]} numberOfLines={1}>
              {selected.subtitle}
            </Text>
          ) : null}
        </View>
        <Text style={[styles.chevron, { color: palette.subtext }]}>v</Text>
      </TouchableOpacity>

      <Modal visible={open} animationType="fade" transparent onRequestClose={() => setOpen(false)}>
        <Pressable style={styles.backdrop} onPress={() => setOpen(false)}>
          <Pressable
            style={[
              styles.sheet,
              {
                backgroundColor: palette.modalBackground,
                borderColor: palette.modalBorder,
              },
            ]}
          >
            <ScrollView showsVerticalScrollIndicator={false}>
              {options.map((option) => {
                const isSelected = option.value === value;
                return (
                  <TouchableOpacity
                    key={option.value}
                    onPress={() => {
                      onChange(option.value);
                      setOpen(false);
                    }}
                    style={[
                      styles.option,
                      isSelected && styles.optionSelected,
                    ]}
                  >
                    <Text
                      style={[
                        styles.optionText,
                        { color: palette.text },
                        isSelected && styles.optionTextSelected,
                      ]}
                    >
                      {option.label}
                    </Text>
                    {option.subtitle ? (
                      <Text style={[styles.optionSubtitle, { color: palette.subtext }]}>
                        {option.subtitle}
                      </Text>
                    ) : null}
                  </TouchableOpacity>
                );
              })}
            </ScrollView>
          </Pressable>
        </Pressable>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    minWidth: 180,
  },
  label: {
    fontSize: 12,
    fontWeight: "600",
    marginBottom: 6,
  },
  trigger: {
    minHeight: 48,
    borderWidth: 1,
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 10,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 10,
  },
  triggerCopy: {
    flex: 1,
  },
  triggerText: {
    fontSize: 14,
    fontWeight: "700",
  },
  triggerSubtitle: {
    marginTop: 2,
    fontSize: 12,
  },
  chevron: {
    fontSize: 14,
    fontWeight: "700",
  },
  backdrop: {
    flex: 1,
    backgroundColor: "rgba(15, 23, 42, 0.45)",
    justifyContent: "center",
    paddingHorizontal: 20,
  },
  sheet: {
    maxHeight: "70%",
    borderRadius: 16,
    borderWidth: 1,
    paddingVertical: 8,
    overflow: "hidden",
  },
  option: {
    paddingHorizontal: 16,
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  optionSelected: {
    backgroundColor: colors.blueTint,
  },
  optionText: {
    fontSize: 15,
    fontWeight: "600",
  },
  optionTextSelected: {
    color: colors.primaryDark,
  },
  optionSubtitle: {
    marginTop: 4,
    fontSize: 12,
    lineHeight: 17,
  },
});
