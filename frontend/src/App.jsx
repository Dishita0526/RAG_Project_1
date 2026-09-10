import { useState, useCallback } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

// ── helpers ──────────────────────────────────────────────────

function basename(path) {
  // Show just the filename portion of a path for readability
  return path ? path.split(/[\\/]/).pop() : path
}

// ── sub-components ───────────────────────────────────────────

function LoadingCard() {
  return (
    <div className="loading-card" role="status" aria-live="polite">
      <div className="loading-card__spinner" aria-hidden="true" />
      <span>Running RAG pipeline&hellip;</span>
      <small style={{ color: 'var(--color-text-muted)', fontSize: 12 }}>
        Embedding → Qdrant search → NVIDIA LLM
      </small>
    </div>
  )
}

function ErrorCard({ message }) {
  return (
    <div className="error-card" role="alert">
      <p className="error-card__title">Something went wrong</p>
      <p>{message}</p>
    </div>
  )
}

function AnswerCard({ answer, sources, numContexts }) {
  return (
    <>
      <div className="answer-card">
        <div className="answer-card__header">
          <span className="answer-card__label">Answer</span>
          <span className="answer-card__context-badge" title="Context chunks retrieved from Qdrant">
            {numContexts} context chunk{numContexts !== 1 ? 's' : ''}
          </span>
        </div>
        <p className="answer-card__text">{answer}</p>
      </div>

      <div className="sources">
        <p className="sources__label">Sources</p>
        {sources.length === 0 ? (
          <p className="sources__empty">No source documents returned.</p>
        ) : (
          <ul className="sources__list" aria-label="Source documents">
            {sources.map((src, i) => (
              <li key={i} className="sources__tag" title={src}>
                📄 {basename(src)}
              </li>
            ))}
          </ul>
        )}
      </div>
    </>
  )
}

// ── IngestPanel ───────────────────────────────────────────────

function IngestPanel() {
  const [path, setPath] = useState('')
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState(null) // { ok: bool, msg: string }

  const handleIngest = useCallback(async () => {
    const trimmed = path.trim()
    if (!trimmed) return

    setLoading(true)
    setStatus(null)

    try {
      const res = await fetch(`${API_URL}/api/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pdf_path: trimmed }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err?.detail || `HTTP ${res.status}`)
      }

      const data = await res.json()
      setStatus({ ok: true, msg: `✓ Ingested ${data.ingested} chunks from "${basename(data.source)}"` })
    } catch (e) {
      setStatus({ ok: false, msg: e.message || 'Ingest failed.' })
    } finally {
      setLoading(false)
    }
  }, [path])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') handleIngest()
  }

  return (
    <div className="ingest-panel">
      <label className="ingest-panel__label" htmlFor="ingest-path">
        Ingest a PDF (absolute path on server)
      </label>
      <div className="ingest-panel__row">
        <input
          id="ingest-path"
          className="ingest-panel__input"
          type="text"
          placeholder="/Users/dishita/RAG_1/AI_Module 1_Chap1.pptx.pdf"
          value={path}
          onChange={(e) => setPath(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
          aria-label="PDF file path"
        />
        <button
          id="ingest-btn"
          className="btn btn--secondary"
          onClick={handleIngest}
          disabled={loading || !path.trim()}
          aria-busy={loading}
        >
          {loading ? (
            <>
              <span className="spinner" style={{ borderTopColor: 'var(--color-accent)', borderColor: 'var(--color-border)' }} aria-hidden="true" />
              Ingesting…
            </>
          ) : (
            'Ingest PDF'
          )}
        </button>
      </div>
      {status && (
        <p className={`ingest-status ${status.ok ? 'ingest-status--success' : 'ingest-status--error'}`}>
          {status.msg}
        </p>
      )}
    </div>
  )
}

// ── App ───────────────────────────────────────────────────────

export default function App() {
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null) // { answer, sources, num_contexts }

  const handleSubmit = useCallback(async () => {
    const trimmed = question.trim()
    if (!trimmed || loading) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await fetch(`${API_URL}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmed, top_k: 5 }),
      })

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}))
        throw new Error(errBody?.detail || `Backend returned HTTP ${res.status}`)
      }

      const data = await res.json()

      if (!data.answer) {
        throw new Error('Unexpected response format from backend.')
      }

      setResult(data)
    } catch (e) {
      if (e.name === 'TypeError' && e.message.includes('fetch')) {
        setError(
          `Cannot reach the backend at ${API_URL}. ` +
          'Make sure "uv run uvicorn main:app" is running.'
        )
      } else {
        setError(e.message || 'An unknown error occurred.')
      }
    } finally {
      setLoading(false)
    }
  }, [question, loading])

  const handleKeyDown = (e) => {
    // Ctrl/Cmd + Enter submits
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      handleSubmit()
    }
  }

  return (
    <main className="app">

      {/* ── Header ── */}
      <header className="header">
        <div className="header__badge">RAG Assistant</div>
        <h1 className="header__title">Ask your documents</h1>
        <p className="header__subtitle">
          Powered by NVIDIA embeddings · Qdrant · LLM
        </p>
      </header>

      {/* ── Ingest panel ── */}
      <IngestPanel />

      <div className="divider" />

      {/* ── Query form ── */}
      <section className="query-form" aria-label="Ask a question">
        <label className="query-form__label" htmlFor="question-input">
          Your question
        </label>
        <textarea
          id="question-input"
          className="query-form__textarea"
          placeholder="What topics are covered in the document?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
          aria-label="Enter your question"
          rows={4}
        />
        <div className="query-form__footer">
          <span className="query-form__hint">⌘ Enter to send</span>
          <button
            id="send-btn"
            className="btn btn--primary"
            onClick={handleSubmit}
            disabled={loading || !question.trim()}
            aria-busy={loading}
          >
            {loading ? (
              <>
                <span className="spinner" aria-hidden="true" />
                Thinking…
              </>
            ) : (
              'Send'
            )}
          </button>
        </div>
      </section>

      {/* ── Loading ── */}
      {loading && <LoadingCard />}

      {/* ── Error ── */}
      {error && !loading && <ErrorCard message={error} />}

      {/* ── Result ── */}
      {result && !loading && (
        <AnswerCard
          answer={result.answer}
          sources={result.sources}
          numContexts={result.num_contexts}
        />
      )}

    </main>
  )
}
