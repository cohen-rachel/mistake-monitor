import React, { useMemo } from "react";
import type { TopicItem } from "../types";
import SelectField from "./SelectField";

interface Props {
  topics: TopicItem[];
  selectedTopicKey: string;
  onSelect: (key: string) => void;
}

export default function TopicPicker({
  topics,
  selectedTopicKey,
  onSelect,
}: Props) {
  const options = useMemo(
    () =>
      topics.map((topic) => ({
        label: topic.title,
        value: topic.key,
        subtitle: topic.prompt,
      })),
    [topics]
  );

  return (
    <SelectField
      label="Practice Topic"
      value={selectedTopicKey}
      options={options}
      onChange={onSelect}
      disabled={topics.length === 0}
    />
  );
}
