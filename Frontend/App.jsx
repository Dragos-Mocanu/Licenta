import React, { useState, useMemo, useRef, useEffect } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { forceManyBody, forceLink, forceCollide } from "d3-force";

export default function App() {
  const [text, setText] = useState("");
  const [tab, setTab] = useState("rake");
  const [results, setResults] = useState({
    rake: [],
    textrank: [],
    relations: [],
    kg: { nodes: [], links: [] },
    qa: { who: [], what: [], where: [], when: [], why: [] },
    ner: {},
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fgRef = useRef();
  const taRef = useRef();

  useEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;
    fg.d3Force("charge", forceManyBody().strength(-200));
    fg.d3Force("link", forceLink().distance(120).id((n) => n.id).strength(1));
    fg.d3Force("collide", forceCollide().radius(30));
    fg.d3AlphaTarget(0.3);
  }, [results.kg]);

  const sendRequest = async (form) => {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        body: form,
      });
      const d = await r.json();
      if (d.error) throw new Error(d.error);
      setText(d.extractedText || "");
      setResults({
        rake: d.rake || [],
        textrank: d.textrank || [],
        relations: d.relations || [],
        kg: d.kg || { nodes: [], links: [] },
        qa: d.qa || { who: [], what: [], where: [], when: [], why: [] },
        ner: d.ner || {},
      });
      setTab("rake");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const form = new FormData();
    form.append("file", f);
    sendRequest(form);
    e.target.value = "";
  };

  const handleSubmit = () => {
    if (!text.trim()) return setError("Introduceţi text mai întâi!");
    const form = new FormData();
    form.append("text", text);
    sendRequest(form);
  };

  const Graph = ({ data }) => (
    <ForceGraph2D
      ref={fgRef}
      graphData={data}
      nodeLabel="id"
      nodeAutoColorBy="id"
      linkDirectionalArrowLength={6}
      linkDirectionalArrowRelPos={1}
      linkCanvasObjectMode={() => "after"}
      linkCanvasObject={(link, ctx, scale) => {
        if (!link.label || link.source.x === undefined) return;
        const fontSize = Math.max(12 / scale, 3);
        ctx.font = `${fontSize}px Sans-Serif`;
        const midX = (link.source.x + link.target.x) / 2;
        const midY = (link.source.y + link.target.y) / 2;
        const w = ctx.measureText(link.label).width;
        ctx.fillStyle = "rgba(255,255,255,0.9)";
        ctx.fillRect(midX - w / 2 - 2, midY - fontSize / 2, w + 4, fontSize + 2);
        ctx.fillStyle = "rgba(0,0,0,0.9)";
        ctx.fillText(link.label, midX, midY);
      }}
    />
  );

  return (
    <div className="container py-5">
      <div className="card shadow p-4">
        <h1 className="text-center text-primary mb-4">
          Information extraction from Romanian texts
        </h1>
        {error && <div className="alert alert-danger">{error}</div>}
        <textarea
          ref={taRef}
          className="form-control mb-3"
          style={{ minHeight: 180 }}
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <div className="d-flex flex-wrap gap-3 mb-3">
          <label className="btn btn-outline-primary mb-0">
            Upload PDF
            <input
              type="file"
              accept="application/pdf"
              hidden
              onChange={handleUpload}
            />
          </label>
          <label className="btn btn-outline-primary mb-0">
            Upload TXT
            <input
              type="file"
              accept="text/plain"
              hidden
              onChange={handleUpload}
            />
          </label>
          <button
            className="btn btn-primary"
            disabled={loading}
            onClick={handleSubmit}
          >
            {loading ? (
              <>
                <span className="spinner-border spinner-border-sm me-2" role="status" />
                Analysing…
              </>
            ) : (
              "Analyze"
            )}
          </button>
        </div>
        <ul className="nav nav-tabs">
          {["rake", "textrank", "ner", "relations", "kg", "qa"].map((t) => (
            <li key={t} className="nav-item">
              <button
                className={`nav-link ${tab === t ? "active" : ""}`}
                onClick={() => setTab(t)}
              >
                {t.toUpperCase()}
              </button>
            </li>
          ))}
        </ul>
        <div className="mt-3">
          {["rake", "textrank"].includes(tab) &&
            (results[tab].length === 0 ? (
              <div className="text-muted">No keywords extracted.</div>
            ) : (
              results[tab].map(({ keyword }, i) => (
                <div key={i} className="card mb-2">
                  <div className="card-body d-flex align-items-center gap-2">
                    <h5 className="card-title mb-0 flex-grow-1">{keyword}</h5>
                  </div>
                </div>
              ))
            ))}
          {tab === "ner" &&
            (() => {
              const nerData = results.ner || {};
              const entries = Object.entries(nerData);
              return entries.length === 0 ? (
                <div className="text-muted">No named entities found.</div>
              ) : (
                entries.map(([label, words], i) => (
                  <div key={i} className="card mb-2">
                    <div className="card-body">
                      <h5 className="card-title">{label}</h5>
                      <ul className="mb-0">
                        {words.map((word, j) => (
                          <li key={j}>{word}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                ))
              );
            })()}
          {tab === "relations" &&
            (results.relations.length === 0 ? (
              <div className="text-muted">No relations extracted.</div>
            ) : (
              results.relations.map((r, i) => (
                <div key={i} className="card mb-2">
                  <div className="card-body">
                    <strong>{r.source}</strong> — <em>{r.label}</em> →{" "}
                    <strong>{r.target}</strong>
                  </div>
                </div>
              ))
            ))}
          {tab === "kg" &&
            (results.kg.nodes.length === 0 ? (
              <div className="text-muted">
                Graph unavailable – run analysis first.
              </div>
            ) : (
              <div style={{ height: 500 }}>
                <Graph data={results.kg} />
              </div>
            ))}
          {tab === "qa" && (
            <div className="row">
              {["who", "what", "where", "when", "why"].map((key) => (
                <div className="col-md-6 mb-3" key={key}>
                  <div className="card">
                    <div className="card-header text-uppercase fw-bold">{key}</div>
                    <div className="card-body">
                      {results.qa[key]?.length === 0 ? (
                        <p className="text-muted mb-0"> Nothing found.</p>
                      ) : (
                        <ul className="mb-0">
                          {results.qa[key].map((v, i) => (
                            <li key={i}>{v}</li>
                          ))}
                        </ul>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}