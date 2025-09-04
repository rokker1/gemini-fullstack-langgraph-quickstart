import type { ReactNode } from "react";
import type { Message } from "@langchain/langgraph-sdk";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Loader2 } from "lucide-react";
import { InputForm } from "@/components/InputForm";

interface ChatMessagesViewProps {
  messages: Message[];
  isLoading: boolean;
  onSubmit: (inputValue: string) => void;
  onCancel: () => void;
}

export function ChatMessagesView({
  messages,
  isLoading,
  onSubmit,
  onCancel,
}: ChatMessagesViewProps) {
  function renderMessageContent(content: Message["content"]): ReactNode {
    if (typeof content === "string") {
      return content;
    }
    if (Array.isArray(content)) {
      return content.map((item, idx) => {
        if (item.type === "text") {
          const textItem = item as { type: "text"; text: unknown };
          const value =
            typeof textItem.text === "string"
              ? textItem.text
              : (textItem.text as any)?.content ?? "";
          return <p key={idx}>{value}</p>;
        }
        return <p key={idx}>{JSON.stringify(item)}</p>;
      });
    }
    return JSON.stringify(content);
  }
  return (
    <div className="flex flex-col h-full">
      <ScrollArea className="flex-1 overflow-y-auto">
        <div className="p-4 md:p-6 space-y-2 max-w-4xl mx-auto pt-16">
          {messages.map((message, index) => (
            <div key={message.id || `msg-${index}`} className="space-y-3">
              <div
                className={`flex items-start gap-3 ${
                  message.type === "human" ? "justify-end" : ""
                }`}
              >
                <div
                  className={`max-w-[75%] rounded-2xl px-4 py-3 ${
                    message.type === "human"
                      ? "bg-blue-600 text-white"
                      : "bg-neutral-700 text-neutral-100"
                  }`}
                >
                  {renderMessageContent(message.content)}
                </div>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-neutral-700 text-neutral-100 px-4 py-3 rounded-2xl flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                Thinking...
              </div>
            </div>
          )}
        </div>
      </ScrollArea>
      <InputForm onSubmit={onSubmit} onCancel={onCancel} isLoading={isLoading} />
    </div>
  );
}
