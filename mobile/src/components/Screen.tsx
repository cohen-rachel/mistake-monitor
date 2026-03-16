import React, { forwardRef, ReactNode } from "react";
import {
  KeyboardAvoidingView,
  NativeSyntheticEvent,
  NativeScrollEvent,
  Platform,
  ScrollView,
  StyleSheet,
  View,
} from "react-native";
import { colors, layout } from "../theme";

const Screen = forwardRef<
  ScrollView,
  {
    children: ReactNode;
    onScroll?: (event: NativeSyntheticEvent<NativeScrollEvent>) => void;
  }
>(function Screen(
  { children, onScroll },
  ref
) {
  return (
    <KeyboardAvoidingView
      style={styles.keyboardArea}
      behavior={Platform.OS === "ios" ? "padding" : "height"}
      keyboardVerticalOffset={Platform.OS === "ios" ? 12 : 0}
    >
      <ScrollView
        ref={ref}
        style={styles.scroll}
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
        keyboardDismissMode="none"
        automaticallyAdjustKeyboardInsets
        onScroll={onScroll}
        scrollEventThrottle={16}
      >
        <View>{children}</View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
});

export default Screen;

const styles = StyleSheet.create({
  keyboardArea: {
    flex: 1,
  },
  scroll: {
    flex: 1,
    backgroundColor: colors.background,
  },
  content: {
    flexGrow: 1,
    padding: layout.screenPadding,
    paddingBottom: 160,
  },
});
