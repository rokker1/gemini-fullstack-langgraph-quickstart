import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message } from "@langchain/langgraph-sdk";
import { useCallback } from "react";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { ChatMessagesView } from "@/components/ChatMessagesView";

export default function App() {
  const thread = useStream<{ messages: Message[]; request: string }>({
    apiUrl: import.meta.env.DEV ? "http://localhost:2024" : "http://localhost:8123",
    assistantId: "agent",
    messagesKey: "messages",
  });

  const handleSubmit = useCallback(
    (submittedInputValue: string) => {
      if (!submittedInputValue.trim()) return;
      const newMessages: Message[] = [
        ...(thread.messages || []),
        { type: "human", content: submittedInputValue, id: Date.now().toString() },
      ];
      thread.submit({ messages: newMessages, request: submittedInputValue });
    },
    [thread]
  );

  const handleCancel = useCallback(() => {
    thread.stop();
    window.location.reload();
  }, [thread]);

  return (
    <div className="flex h-screen bg-neutral-800 text-neutral-100 font-sans antialiased">
      <main className="h-full w-full max-w-4xl mx-auto">
        {thread.messages.length === 0 ? (
          <WelcomeScreen
            handleSubmit={handleSubmit}
            isLoading={thread.isLoading}
            onCancel={handleCancel}
          />
        ) : (
          <ChatMessagesView
            messages={thread.messages}
            isLoading={thread.isLoading}
            onSubmit={handleSubmit}
            onCancel={handleCancel}
          />
        )}
      </main>
    </div>
  );
}
