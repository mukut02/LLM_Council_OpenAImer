import { useEffect, useMemo, useState } from "react";

function MetricCard({ label, value }) {
  const display = Number.isFinite(value) ? value.toFixed(4) : "-";
  return (
    <div className="metric-card">
      <p className="metric-label">{label}</p>
      <p className="metric-value">{display}</p>
    </div>
  );
}

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [modelAnswer, setModelAnswer] = useState("");
  const [numExamples, setNumExamples] = useState(3);
  const [availableApiSlots, setAvailableApiSlots] = useState(["api1", "api2", "api3"]);
  const [selectedApiIndex, setSelectedApiIndex] = useState(0);
  const [outputs, setOutputs] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [metrics, setMetrics] = useState(null);
  const [loadingGenerate, setLoadingGenerate] = useState(false);
  const [loadingEvaluate, setLoadingEvaluate] = useState(false);
  const [error, setError] = useState("");

  const selectedOutput = useMemo(
    () => outputs.find((item) => item.model === selectedModel),
    [outputs, selectedModel]
  );
  const selectedApiSlot = availableApiSlots[selectedApiIndex] ?? availableApiSlots[0] ?? "api1";

  async function loadApiSlots() {
    try {
      const res = await fetch("/api/api-slots");
      if (!res.ok) {
        return;
      }

      const data = await res.json();
      const nextSlots = data.available_api_slots ?? [];
      if (!nextSlots.length) {
        return;
      }

      setAvailableApiSlots(nextSlots);
      const preferredSlot = data.default_api_slot ?? nextSlots[0];
      const nextIndex = nextSlots.indexOf(preferredSlot);
      setSelectedApiIndex(nextIndex >= 0 ? nextIndex : 0);
    } catch {
      // Keep the local fallback list if the API-slot probe fails.
    }
  }

  useEffect(() => {
    loadApiSlots();
  }, []);

  async function handleGenerate() {
    setError("");
    setMetrics(null);

    if (!prompt.trim()) {
      setError("Please enter a prompt first.");
      return;
    }

    try {
      setLoadingGenerate(true);
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          num_examples: numExamples,
          api_slot: selectedApiSlot
        })
      });

      if (!res.ok) {
        throw new Error("Generation failed.");
      }

      const data = await res.json();
      const nextOutputs = data.outputs ?? [];
      setOutputs(nextOutputs);
      setSelectedModel(nextOutputs[0]?.model ?? "");
    } catch (e) {
      setError(e.message || "Something went wrong during generation.");
    } finally {
      setLoadingGenerate(false);
    }
  }

  async function handleEvaluate() {
    setError("");

    if (!prompt.trim() || !modelAnswer.trim()) {
      setError("Please provide both prompt and model answer.");
      return;
    }

    try {
      setLoadingEvaluate(true);
      const res = await fetch("/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, model_answer: modelAnswer })
      });

      if (!res.ok) {
        throw new Error("Evaluation failed.");
      }

      const data = await res.json();
      setMetrics(data);
    } catch (e) {
      setError(e.message || "Something went wrong during evaluation.");
    } finally {
      setLoadingEvaluate(false);
    }
  }

  return (
    <div className="page">
      <header className="hero">
        <p className="eyebrow">MENTAL AI</p>
        <h1>Prompt and Model Answer Evaluator</h1>
        <p className="sub">
          Generate few-shot model responses, pick one, and evaluate quality in one clean flow.
        </p>
      </header>

      <main className="layout">
        <section className="panel">
          <div className="field">
            <span>Active API</span>
            <div className="api-slider-card">
              <input
                type="range"
                min={0}
                max={Math.max(availableApiSlots.length - 1, 0)}
                step={1}
                value={selectedApiIndex}
                onChange={(e) => setSelectedApiIndex(Number(e.target.value))}
                disabled={availableApiSlots.length <= 1}
              />
              <div className="api-slider-meta">
                <strong>{selectedApiSlot.toUpperCase()}</strong>
                <span>
                  {availableApiSlots.length} available
                  {availableApiSlots.length === 1 ? " slot" : " slots"}
                </span>
              </div>
              <div className="api-slot-list">
                {availableApiSlots.map((slot, index) => (
                  <button
                    key={slot}
                    type="button"
                    className={`slot-chip ${index === selectedApiIndex ? "active" : ""}`}
                    onClick={() => setSelectedApiIndex(index)}
                  >
                    {slot.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <label className="field">
            <span>Enter Prompt</span>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe the user message or question..."
              rows={6}
            />
          </label>

          <label className="field">
            <span>Enter Model Answer</span>
            <textarea
              value={modelAnswer}
              onChange={(e) => setModelAnswer(e.target.value)}
              placeholder="Paste or generate a model response..."
              rows={8}
            />
          </label>

          <div className="actions">
            <button
              className="btn btn-primary"
              onClick={handleEvaluate}
              disabled={loadingEvaluate}
            >
              {loadingEvaluate ? "Evaluating..." : "Evaluate"}
            </button>
          </div>
        </section>

        <section className="panel">
          <div className="row">
            <label className="field compact">
              <span>Few-shot examples</span>
              <input
                type="number"
                min={1}
                max={10}
                value={numExamples}
                onChange={(e) => setNumExamples(Number(e.target.value))}
              />
            </label>

            <button
              className="btn btn-secondary"
              onClick={handleGenerate}
              disabled={loadingGenerate}
            >
              {loadingGenerate ? "Generating..." : "Generate Answers"}
            </button>
          </div>

          {outputs.length > 0 && (
            <>
              <label className="field compact">
                <span>Generated model</span>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                >
                  {outputs.map((o) => (
                    <option key={o.model} value={o.model}>
                      {o.model}
                    </option>
                  ))}
                </select>
              </label>

              <div className="output-card">
                <p>{selectedOutput?.text || "No output selected."}</p>
              </div>

              <button
                className="btn btn-ghost"
                onClick={() => setModelAnswer(selectedOutput?.text || "")}
              >
                Use Selected Output
              </button>
            </>
          )}

          {metrics && (
            <div className="metrics-grid">
              <MetricCard
                label="Semantic Similarity"
                value={metrics.semantic_similarity}
              />
              <MetricCard label="BERTScore (F1)" value={metrics.bertscore_f1} />
              <MetricCard label="Weighted Score" value={metrics.weighted_score} />
            </div>
          )}
        </section>
      </main>

      {error && <p className="error">{error}</p>}
    </div>
  );
}
