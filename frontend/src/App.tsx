import { useCallback, useState } from "react";
import { InputForm } from "@/components/InputForm";

interface Artifacts {
  json?: string;
  puml?: string;
  sql?: string;
}

const API_URL = import.meta.env.DEV ? "http://localhost:2024" : "";

export default function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [artifacts, setArtifacts] = useState<Artifacts | null>(null);

  const handleSubmit = useCallback(async (query: string) => {
    setIsLoading(true);
    setArtifacts(null);
    try {
      const res = await fetch(`${API_URL}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();
      setArtifacts(data.artifacts || data);
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleCancel = useCallback(() => {
    setArtifacts(null);
  }, []);

  return (
    <div className="flex h-screen bg-neutral-800 text-neutral-100 font-sans antialiased">
      <main className="h-full w-full max-w-4xl mx-auto">
        <div className="p-4 md:p-6 space-y-4">
          <InputForm onSubmit={handleSubmit} onCancel={handleCancel} isLoading={isLoading} />
          {artifacts && (
            <div className="space-y-4">
              {artifacts.json && (
                <pre className="whitespace-pre-wrap bg-neutral-900 p-4 rounded">{artifacts.json}</pre>
              )}
              {artifacts.sql && (
                <pre className="whitespace-pre-wrap bg-neutral-900 p-4 rounded">{artifacts.sql}</pre>
              )}
              {artifacts.puml && (
                <pre className="whitespace-pre-wrap bg-neutral-900 p-4 rounded">{artifacts.puml}</pre>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

