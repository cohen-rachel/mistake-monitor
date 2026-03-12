import React, { useEffect, useRef, useState } from "react";
import { Audio } from "expo-av";
import PrimaryButton from "./PrimaryButton";
import type { MobileAudioFile } from "../types";

interface Props {
  onStatusChange: (recording: boolean) => void;
  onRecordingReady: (file: MobileAudioFile | null) => void;
}

export default function AudioRecorder({
  onStatusChange,
  onRecordingReady,
}: Props) {
  const [isRecording, setIsRecording] = useState(false);
  const recordingRef = useRef<Audio.Recording | null>(null);

  useEffect(() => {
    return () => {
      if (recordingRef.current) {
        void recordingRef.current.stopAndUnloadAsync().catch(() => undefined);
      }
    };
  }, []);

  const startRecording = async () => {
    const permission = await Audio.requestPermissionsAsync();
    if (!permission.granted) {
      throw new Error("Microphone permission is required to record audio.");
    }

    await Audio.setAudioModeAsync({
      allowsRecordingIOS: true,
      playsInSilentModeIOS: true,
    });

    const { recording } = await Audio.Recording.createAsync(
      Audio.RecordingOptionsPresets.HIGH_QUALITY
    );
    recordingRef.current = recording;
    setIsRecording(true);
    onStatusChange(true);
    onRecordingReady(null);
  };

  const stopRecording = async () => {
    const recording = recordingRef.current;
    if (!recording) {
      return;
    }
    await recording.stopAndUnloadAsync();
    const uri = recording.getURI();
    recordingRef.current = null;
    setIsRecording(false);
    onStatusChange(false);
    if (!uri) {
      onRecordingReady(null);
      return;
    }
    onRecordingReady({
      uri,
      name: `recording-${Date.now()}.m4a`,
      type: "audio/m4a",
    });
  };

  return (
    <PrimaryButton
      label={isRecording ? "Stop Recording" : "Start Recording"}
      onPress={() => {
        void (async () => {
          try {
            await (isRecording ? stopRecording() : startRecording());
          } catch (error) {
            console.warn("Recording failed", error);
            setIsRecording(false);
            onStatusChange(false);
            onRecordingReady(null);
          }
        })();
      }}
      tone={isRecording ? "danger" : "primary"}
    />
  );
}
