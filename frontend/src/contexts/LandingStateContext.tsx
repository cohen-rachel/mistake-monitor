import React, { createContext, ReactNode, useContext, useState } from "react";
import type { MistakeOut, TopicAttemptItem, TopicItem } from "../types";

type LandingTab = "record" | "upload";

interface LandingStateContextType {
  tab: LandingTab;
  setTab: React.Dispatch<React.SetStateAction<LandingTab>>;
  topics: TopicItem[];
  setTopics: React.Dispatch<React.SetStateAction<TopicItem[]>>;
  estimatedLevel: string;
  setEstimatedLevel: React.Dispatch<React.SetStateAction<string>>;
  selectedTopicKey: string;
  setSelectedTopicKey: React.Dispatch<React.SetStateAction<string>>;
  topicHistory: TopicAttemptItem[];
  setTopicHistory: React.Dispatch<React.SetStateAction<TopicAttemptItem[]>>;
  topicsLoading: boolean;
  setTopicsLoading: React.Dispatch<React.SetStateAction<boolean>>;
  isRecording: boolean;
  setIsRecording: React.Dispatch<React.SetStateAction<boolean>>;
  elapsedSec: number;
  setElapsedSec: React.Dispatch<React.SetStateAction<number>>;
  liveTranscript: string;
  setLiveTranscript: React.Dispatch<React.SetStateAction<string>>;
  transcriptAnalyzed: boolean;
  setTranscriptAnalyzed: React.Dispatch<React.SetStateAction<boolean>>;
  analyzing: boolean;
  setAnalyzing: React.Dispatch<React.SetStateAction<boolean>>;
  mistakes: MistakeOut[];
  setMistakes: React.Dispatch<React.SetStateAction<MistakeOut[]>>;
  statusMsg: string;
  setStatusMsg: React.Dispatch<React.SetStateAction<string>>;
  uploadFile: File | null;
  setUploadFile: React.Dispatch<React.SetStateAction<File | null>>;
  uploadAnalyzing: boolean;
  setUploadAnalyzing: React.Dispatch<React.SetStateAction<boolean>>;
  uploadMistakes: MistakeOut[];
  setUploadMistakes: React.Dispatch<React.SetStateAction<MistakeOut[]>>;
  uploadTranscript: string;
  setUploadTranscript: React.Dispatch<React.SetStateAction<string>>;
  uploadStatus: string;
  setUploadStatus: React.Dispatch<React.SetStateAction<string>>;
}

const LandingStateContext = createContext<LandingStateContextType | undefined>(
  undefined
);

export const LandingStateProvider = ({ children }: { children: ReactNode }) => {
  const [tab, setTab] = useState<LandingTab>("record");
  const [topics, setTopics] = useState<TopicItem[]>([]);
  const [estimatedLevel, setEstimatedLevel] = useState("beginner");
  const [selectedTopicKey, setSelectedTopicKey] = useState("free_talk");
  const [topicHistory, setTopicHistory] = useState<TopicAttemptItem[]>([]);
  const [topicsLoading, setTopicsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [elapsedSec, setElapsedSec] = useState(0);
  const [liveTranscript, setLiveTranscript] = useState("");
  const [transcriptAnalyzed, setTranscriptAnalyzed] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [mistakes, setMistakes] = useState<MistakeOut[]>([]);
  const [statusMsg, setStatusMsg] = useState("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadAnalyzing, setUploadAnalyzing] = useState(false);
  const [uploadMistakes, setUploadMistakes] = useState<MistakeOut[]>([]);
  const [uploadTranscript, setUploadTranscript] = useState("");
  const [uploadStatus, setUploadStatus] = useState("");

  return (
    <LandingStateContext.Provider
      value={{
        tab,
        setTab,
        topics,
        setTopics,
        estimatedLevel,
        setEstimatedLevel,
        selectedTopicKey,
        setSelectedTopicKey,
        topicHistory,
        setTopicHistory,
        topicsLoading,
        setTopicsLoading,
        isRecording,
        setIsRecording,
        elapsedSec,
        setElapsedSec,
        liveTranscript,
        setLiveTranscript,
        transcriptAnalyzed,
        setTranscriptAnalyzed,
        analyzing,
        setAnalyzing,
        mistakes,
        setMistakes,
        statusMsg,
        setStatusMsg,
        uploadFile,
        setUploadFile,
        uploadAnalyzing,
        setUploadAnalyzing,
        uploadMistakes,
        setUploadMistakes,
        uploadTranscript,
        setUploadTranscript,
        uploadStatus,
        setUploadStatus,
      }}
    >
      {children}
    </LandingStateContext.Provider>
  );
};

export const useLandingState = () => {
  const context = useContext(LandingStateContext);
  if (!context) {
    throw new Error("useLandingState must be used within LandingStateProvider");
  }
  return context;
};
