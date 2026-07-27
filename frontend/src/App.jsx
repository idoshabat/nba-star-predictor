import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Loader2,
  Play,
  RotateCcw,
  Sparkles,
} from 'lucide-react'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

const FIELD_GROUPS = [
  {
    title: 'Profile',
    fields: [
      { key: 'age', label: 'Age', step: 1 },
      { key: 'gp', label: 'Games', step: 1 },
      { key: 'min', label: 'Minutes', step: 1 },
    ],
  },
  {
    title: 'Production',
    fields: [
      { key: 'pts', label: 'Points', step: 1 },
      { key: 'reb', label: 'Rebounds', step: 1 },
      { key: 'ast', label: 'Assists', step: 1 },
      { key: 'stl', label: 'Steals', step: 1 },
      { key: 'blk', label: 'Blocks', step: 1 },
    ],
  },
  {
    title: 'Shooting',
    fields: [
      { key: 'fg_pct', label: 'FG%', step: 0.001 },
      { key: 'fg3_pct', label: '3P%', step: 0.001 },
      { key: 'ft_pct', label: 'FT%', step: 0.001 },
    ],
  },
]

const EMPTY_STATS = {
  age: 20,
  gp: 60,
  pts: 600,
  reb: 220,
  ast: 140,
  stl: 45,
  blk: 25,
  min: 1500,
  fg_pct: 0.45,
  fg3_pct: 0.34,
  ft_pct: 0.75,
}

function App() {
  const [examples, setExamples] = useState([])
  const [selectedExample, setSelectedExample] = useState('')
  const [formData, setFormData] = useState(EMPTY_STATS)
  const [prediction, setPrediction] = useState(null)
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState('')

  useEffect(() => {
    async function loadExamples() {
      try {
        const response = await fetch(`${API_BASE_URL}/examples`)

        if (!response.ok) {
          throw new Error('Could not load examples')
        }

        const data = await response.json()
        setExamples(data)

        if (data.length > 0) {
          setSelectedExample(data[0].name)
          setFormData(data[0].stats)
        }
      } catch {
        setError('Start the FastAPI backend and refresh the dashboard.')
      }
    }

    loadExamples()
  }, [])

  const probabilityPercent = useMemo(() => {
    if (!prediction) {
      return 0
    }

    return Math.round(prediction.probability * 100)
  }, [prediction])

  function handleExampleChange(event) {
    const name = event.target.value
    const example = examples.find((item) => item.name === name)

    setSelectedExample(name)
    setPrediction(null)
    setError('')

    if (example) {
      setFormData(example.stats)
    }
  }

  function handleFieldChange(key, value) {
    setFormData((current) => ({
      ...current,
      [key]: Number(value),
    }))
    setPrediction(null)
  }

  async function handlePredict(event) {
    event.preventDefault()
    setStatus('loading')
    setError('')

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      })

      if (!response.ok) {
        throw new Error('Prediction request failed')
      }

      const data = await response.json()
      setPrediction(data)
      setStatus('idle')
    } catch {
      setStatus('idle')
      setError('Prediction failed. Make sure the FastAPI backend is running.')
    }
  }

  function resetForm() {
    const example = examples.find((item) => item.name === selectedExample)

    setPrediction(null)
    setError('')
    setFormData(example?.stats ?? EMPTY_STATS)
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">NBA Future Star Predictor</p>
          <h1>Rookie signal dashboard</h1>
        </div>
        <div className="api-pill">
          <Activity size={16} aria-hidden="true" />
          FastAPI + XGBoost
        </div>
      </header>

      <section className="workspace">
        <form className="panel input-panel" onSubmit={handlePredict}>
          <div className="panel-header">
            <div>
              <p className="section-label">Player Input</p>
              <h2>Rookie season stats</h2>
            </div>
            <button className="icon-button" type="button" onClick={resetForm} title="Reset stats">
              <RotateCcw size={18} aria-hidden="true" />
            </button>
          </div>

          <label className="select-label" htmlFor="example-player">
            Demo player
          </label>
          <select id="example-player" value={selectedExample} onChange={handleExampleChange}>
            {examples.map((example) => (
              <option key={example.name} value={example.name}>
                {example.name}
              </option>
            ))}
          </select>

          {FIELD_GROUPS.map((group) => (
            <fieldset key={group.title} className="field-group">
              <legend>{group.title}</legend>
              <div className="field-grid">
                {group.fields.map((field) => (
                  <label key={field.key} className="stat-field">
                    <span>{field.label}</span>
                    <input
                      type="number"
                      min="0"
                      max={field.key.includes('pct') ? 1 : undefined}
                      step={field.step}
                      value={formData[field.key]}
                      onChange={(event) => handleFieldChange(field.key, event.target.value)}
                    />
                  </label>
                ))}
              </div>
            </fieldset>
          ))}

          <button className="primary-button" type="submit" disabled={status === 'loading'}>
            {status === 'loading' ? (
              <Loader2 className="spin" size={18} aria-hidden="true" />
            ) : (
              <Play size={18} aria-hidden="true" />
            )}
            Predict
          </button>
        </form>

        <section className="panel result-panel" aria-live="polite">
          <div className="panel-header">
            <div>
              <p className="section-label">Model Output</p>
              <h2>All-Star probability</h2>
            </div>
            <Sparkles size={22} aria-hidden="true" />
          </div>

          {error ? (
            <div className="notice error">
              <AlertCircle size={20} aria-hidden="true" />
              {error}
            </div>
          ) : null}

          <div className="score-block">
            <div className="score-ring" style={{ '--score': `${probabilityPercent}%` }}>
              <span>{prediction ? probabilityPercent : '--'}%</span>
            </div>
            <div>
              <p className="prediction-label">
                {prediction?.prediction ?? 'Run a prediction'}
              </p>
              <p className="muted">
                Threshold {prediction?.threshold ?? 0.45} · XGBoost classifier
              </p>
            </div>
          </div>

          <div className="metrics-row">
            <div>
              <span>F1</span>
              <strong>0.577</strong>
            </div>
            <div>
              <span>Precision</span>
              <strong>0.533</strong>
            </div>
            <div>
              <span>Recall</span>
              <strong>0.629</strong>
            </div>
          </div>

          <div className="signals">
            <div className="subhead">
              <BarChart3 size={18} aria-hidden="true" />
              Signals
            </div>
            <ul>
              {(prediction?.signals ?? ['Choose a player and run the model']).map((signal) => (
                <li key={signal}>
                  <CheckCircle2 size={17} aria-hidden="true" />
                  {signal}
                </li>
              ))}
            </ul>
          </div>
        </section>
      </section>
    </main>
  )
}

export default App
