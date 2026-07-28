import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity,
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Loader2,
  Play,
  RotateCcw,
  Search,
  Sparkles,
  Trophy,
} from 'lucide-react'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'https://nba-star-predictor-api.onrender.com'
const NBA_LOGO_URL = 'https://cdn.nba.com/logos/leagues/logo-nba.svg'
const ROOKIE_RANKING_SEASON = '2025-26'

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

const DISPLAY_STATS = [
  { key: 'ppg', label: 'PPG' },
  { key: 'rpg', label: 'RPG' },
  { key: 'apg', label: 'APG' },
  { key: 'spg', label: 'SPG' },
  { key: 'bpg', label: 'BPG' },
  { key: 'mpg', label: 'MPG' },
]

const LOW_CHANCE_RULES = [
  {
    test: (features) => features.ppg < 8,
    note: 'Low scoring volume compared with stronger future All-Star rookie profiles',
  },
  {
    test: (features) => features.mpg < 18,
    note: 'Limited minutes suggest a smaller season role',
  },
  {
    test: (features) => features.apg < 2.5 && features.rpg < 4,
    note: 'Few secondary production signals from passing or rebounding',
  },
  {
    test: (features) => features.efficiency < 0.45,
    note: 'Box-score efficiency is below the model’s stronger prospect patterns',
  },
  {
    test: (features) => features.age > 22,
    note: 'Older rookie age gives the model less long-term development signal',
  },
]

function App() {
  const resultPanelRef = useRef(null)
  const [examples, setExamples] = useState([])
  const [selectedExample, setSelectedExample] = useState('')
  const [formData, setFormData] = useState(EMPTY_STATS)
  const [prediction, setPrediction] = useState(null)
  const [status, setStatus] = useState('idle')
  const [playerSearchStatus, setPlayerSearchStatus] = useState('idle')
  const [playerQuery, setPlayerQuery] = useState('')
  const [playerResults, setPlayerResults] = useState([])
  const [rookieRankingStatus, setRookieRankingStatus] = useState('idle')
  const [rookieRankings, setRookieRankings] = useState([])
  const [seasonMode, setSeasonMode] = useState('rookie')
  const [selectedActivePlayer, setSelectedActivePlayer] = useState(null)
  const [selectedPlayerContext, setSelectedPlayerContext] = useState('')
  const [error, setError] = useState('')
  const statsLocked = Boolean(selectedActivePlayer || selectedExample)

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
          setSelectedPlayerContext(`Demo player · ${data[0].name}`)
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

  const displayStats = useMemo(() => {
    const games = Number(formData.gp)
    const safePerGame = (value) => {
      if (!games) {
        return '0.0'
      }

      return (Number(value) / games).toFixed(1)
    }

    return {
      ppg: safePerGame(formData.pts),
      rpg: safePerGame(formData.reb),
      apg: safePerGame(formData.ast),
      spg: safePerGame(formData.stl),
      bpg: safePerGame(formData.blk),
      mpg: safePerGame(formData.min),
    }
  }, [formData])

  const lowChanceNotes = useMemo(() => {
    if (!prediction || prediction.probability >= prediction.threshold) {
      return []
    }

    const features = prediction.features_used ?? {}
    const notes = LOW_CHANCE_RULES.filter((rule) => rule.test(features)).map((rule) => rule.note)

    if (prediction.season_mode === 'latest') {
      notes.push('Latest-season mode is exploratory; the model was trained on rookie-season profiles')
    }

    return notes.length > 0
      ? notes.slice(0, 4)
      : ['The player does not cross enough high-impact thresholds for this model']
  }, [prediction])

  function handleExampleChange(event) {
    const name = event.target.value
    const example = examples.find((item) => item.name === name)

    setSelectedExample(name)
    setSelectedActivePlayer(null)
    setPrediction(null)
    setError('')
    setSelectedPlayerContext(example ? `Demo player · ${name}` : '')

    if (example) {
      loadExamplePlayer(example)
    } else {
      setFormData(EMPTY_STATS)
    }
  }

  function handleFieldChange(key, value) {
    setFormData((current) => ({
      ...current,
      [key]: Number(value),
    }))
    setPrediction(null)
  }

  function seasonModeLabel(mode) {
    return mode === 'rookie' ? 'rookie season' : 'latest available season'
  }

  function actualOutcomeText(value) {
    if (value === true) {
      return 'Yes'
    }

    if (value === false) {
      return 'No'
    }

    return 'TBD'
  }

  function actualOutcomeClass(value) {
    if (value === true) {
      return 'actual-badge positive'
    }

    if (value === false) {
      return 'actual-badge neutral'
    }

    return 'actual-badge pending'
  }

  function scrollToResultPanel() {
    window.requestAnimationFrame(() => {
      resultPanelRef.current?.scrollIntoView({
        behavior: 'smooth',
        block: 'start',
      })
    })
  }

  async function readApiError(response, fallbackMessage) {
    try {
      const data = await response.json()
      return data.detail ?? fallbackMessage
    } catch {
      return fallbackMessage
    }
  }

  async function loadExamplePlayer(example, mode = seasonMode) {
    setStatus('loading')
    setError('')
    setSelectedActivePlayer(null)
    setSelectedPlayerContext(`Demo player · ${example.name}`)

    try {
      const response = await fetch(
        `${API_BASE_URL}/players/${example.player_id}/prediction?season_mode=${mode}`,
      )

      if (!response.ok) {
        throw new Error(await readApiError(response, 'Demo prediction failed'))
      }

      const data = await response.json()
      setFormData(data.stats)
      setPrediction(data)
      setStatus('idle')
    } catch (caughtError) {
      setFormData(example.stats)
      setPrediction(null)
      setStatus('idle')
      setError(caughtError.message)
    }
  }

  async function handlePlayerSearch() {
    if (playerQuery.trim().length < 2) {
      setPlayerResults([])
      return
    }

    setPlayerSearchStatus('loading')
    setError('')

    try {
      const response = await fetch(
        `${API_BASE_URL}/players/search?query=${encodeURIComponent(playerQuery)}&limit=8`,
      )

      if (!response.ok) {
        throw new Error('Search failed')
      }

      const data = await response.json()
      setPlayerResults(data)
      setPlayerSearchStatus('idle')
    } catch {
      setPlayerSearchStatus('idle')
      setError('Player search failed. Make sure the FastAPI backend is running.')
    }
  }

  async function loadCurrentPlayer(player, mode = seasonMode) {
    setStatus('loading')
    setError('')
    setSelectedExample('')
    setSelectedActivePlayer(player)
    setSelectedPlayerContext(`Active player · ${player.name}`)

    try {
      const response = await fetch(
        `${API_BASE_URL}/players/${player.player_id}/prediction?season_mode=${mode}`,
      )

      if (!response.ok) {
        throw new Error(await readApiError(response, 'Player prediction failed'))
      }

      const data = await response.json()
      setFormData(data.stats)
      setPrediction(data)
      setStatus('idle')
      setPlayerResults([])
      setPlayerQuery(player.name)
    } catch (caughtError) {
      setStatus('idle')
      setError(caughtError.message)
    }
  }

  async function loadRookieRankings() {
    setRookieRankingStatus('loading')
    setError('')

    try {
      const response = await fetch(
        `${API_BASE_URL}/rookies/rankings?season=${ROOKIE_RANKING_SEASON}&limit=5&min_games=20`,
      )

      if (!response.ok) {
        throw new Error(await readApiError(response, 'Rookie ranking failed'))
      }

      const data = await response.json()
      setRookieRankings(data)
      setRookieRankingStatus('idle')
    } catch (caughtError) {
      setRookieRankingStatus('idle')
      setError(caughtError.message)
    }
  }

  function selectRookieRanking(player) {
    setSelectedExample('')
    setSelectedActivePlayer({
      player_id: player.player_id,
      name: player.player_name,
    })
    setSelectedPlayerContext(`Rookie board · ${player.player_name}`)
    setPlayerQuery(player.player_name)
    setFormData(player.stats)
    setPrediction(player)
    setError('')
  }

  function handleSeasonModeChange(mode) {
    setSeasonMode(mode)

    if (selectedActivePlayer) {
      loadCurrentPlayer(selectedActivePlayer, mode)
      return
    }

    if (selectedExample) {
      const example = examples.find((item) => item.name === selectedExample)

      if (example) {
        loadExamplePlayer(example, mode)
      }
    }
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
        throw new Error(await readApiError(response, 'Prediction request failed'))
      }

      const data = await response.json()
      setPrediction(data)
      setStatus('idle')
      scrollToResultPanel()
    } catch (caughtError) {
      setStatus('idle')
      setError(caughtError.message)
      scrollToResultPanel()
    }
  }

  function resetForm() {
    const example = examples.find((item) => item.name === selectedExample)

    setPrediction(null)
    setError('')
    setSelectedActivePlayer(null)

    if (example) {
      loadExamplePlayer(example)
    } else {
      setFormData(EMPTY_STATS)
      setSelectedPlayerContext('')
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <img className="nba-logo" src={NBA_LOGO_URL} alt="NBA logo" />
          <div>
            <p className="eyebrow">NBA Future Star Predictor</p>
            <h1>Future All-Star dashboard</h1>
          </div>
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
              <h2>Season stats</h2>
              <p className="section-description">
                Search a player, choose a season view, and review the real stats sent to the model.
              </p>
            </div>
            <button className="icon-button" type="button" onClick={resetForm} title="Reset stats">
              <RotateCcw size={18} aria-hidden="true" />
            </button>
          </div>

          <div className="lookup-block">
            <div className="search-row">
              <div className="lookup-label-row">
                <label className="select-label" htmlFor="player-search">
                  Search active player
                </label>
                <div className="mode-toggle" aria-label="Season mode">
                  <button
                    type="button"
                    className={seasonMode === 'rookie' ? 'mode-button active' : 'mode-button'}
                    onClick={() => handleSeasonModeChange('rookie')}
                  >
                    Rookie
                  </button>
                  <button
                    type="button"
                    className={seasonMode === 'latest' ? 'mode-button active' : 'mode-button'}
                    onClick={() => handleSeasonModeChange('latest')}
                  >
                    Latest
                  </button>
                </div>
              </div>
              <div className="search-controls">
                <input
                  id="player-search"
                  type="search"
                  value={playerQuery}
                  placeholder="Try Victor Wembanyama"
                  onChange={(event) => setPlayerQuery(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                      event.preventDefault()
                      handlePlayerSearch()
                    }
                  }}
                />
                <button
                  className="secondary-button"
                  type="button"
                  onClick={handlePlayerSearch}
                  disabled={playerSearchStatus === 'loading'}
                  title="Search active players"
                >
                  {playerSearchStatus === 'loading' ? (
                    <Loader2 className="spin" size={17} aria-hidden="true" />
                  ) : (
                    <Search size={17} aria-hidden="true" />
                  )}
                </button>
              </div>
            </div>

            {playerResults.length > 0 ? (
              <div className="search-results">
                {playerResults.map((player) => (
                  <button
                    key={player.player_id}
                    type="button"
                    onClick={() => loadCurrentPlayer(player)}
                  >
                    {player.name}
                  </button>
                ))}
              </div>
            ) : null}

            <div className="divider-label">or use a saved demo</div>

            <label className="select-label" htmlFor="example-player">
              Demo player
            </label>
            <select id="example-player" value={selectedExample} onChange={handleExampleChange}>
              <option value="">Choose demo player</option>
              {examples.map((example) => (
                <option key={example.name} value={example.name}>
                  {example.name}
                </option>
              ))}
            </select>
          </div>

          <div className="display-stats" aria-label="Per-game stats">
            {DISPLAY_STATS.map((stat) => (
              <div key={stat.key}>
                <span>{stat.label}</span>
                <strong>{displayStats[stat.key]}</strong>
              </div>
            ))}
          </div>

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
                      disabled={statsLocked}
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

        <section ref={resultPanelRef} className="panel result-panel" aria-live="polite">
          <div className="panel-header">
            <div>
              <p className="section-label">Model Output</p>
              <h2>All-Star probability</h2>
              <p className="section-description">
                The XGBoost model returns a probability, decision label, and readable signals.
              </p>
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
              {prediction?.season ? (
                <p className="muted context-line">
                  {prediction.player_name} · {seasonModeLabel(prediction.season_mode)}{' '}
                  {prediction.season}
                </p>
              ) : selectedPlayerContext ? (
                <p className="muted context-line">{selectedPlayerContext}</p>
              ) : null}
              {prediction ? (
                <div className={actualOutcomeClass(prediction.actual_all_star)}>
                  Ever All-Star: {actualOutcomeText(prediction.actual_all_star)}
                  {prediction.actual_all_star_seasons?.length
                    ? ` · ${prediction.actual_all_star_seasons.join(', ')}`
                    : ''}
                </div>
              ) : null}
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

          {lowChanceNotes.length > 0 ? (
            <div className="model-notes">
              <div className="subhead">
                <AlertCircle size={18} aria-hidden="true" />
                Model notes
              </div>
              <ul>
                {lowChanceNotes.map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </section>
      </section>

      <section className="panel rookie-panel">
        <div className="panel-header rookie-panel-header">
          <div>
            <p className="section-label">Rookie Board</p>
            <h2>Top 5 rookie All-Star projections</h2>
            <p className="section-description">
              Ranks last season's rookies by their model probability using NBA API season totals.
            </p>
          </div>
          <button
            className="rookie-load-button"
            type="button"
            onClick={loadRookieRankings}
            disabled={rookieRankingStatus === 'loading'}
          >
            {rookieRankingStatus === 'loading' ? (
              <Loader2 className="spin" size={17} aria-hidden="true" />
            ) : null}
            Load rankings
          </button>
        </div>

        {rookieRankings.length > 0 ? (
          <div className="rookie-board">
            <ol>
              {rookieRankings.map((player) => (
                <li key={player.player_id}>
                  <button type="button" onClick={() => selectRookieRanking(player)}>
                    <span className="rank-number">#{player.rank}</span>
                    <span className="rank-player">
                      <strong>{player.player_name}</strong>
                      <small>
                        {player.team_abbreviation} · {player.season}
                      </small>
                    </span>
                    <span className="rank-score">{Math.round(player.probability * 100)}%</span>
                  </button>
                </li>
              ))}
            </ol>
          </div>
        ) : (
          <div className="rookie-empty-state">
            <Trophy size={18} aria-hidden="true" />
            Load the latest rookie ranking to compare the strongest rookie profiles.
          </div>
        )}
      </section>

      <section className="panel model-card-panel">
        <div className="panel-header">
          <div>
            <p className="section-label">Model Card</p>
            <h2>How the model performs</h2>
            <p className="section-description">
              These metrics describe the trained XGBoost classifier, separate from any single player
              prediction.
            </p>
            <p className="model-card-note">
              Accuracy is less useful here because the dataset is imbalanced: most NBA players never
              become All-Stars, so a model can look accurate while missing the rare future stars.
            </p>
          </div>
          <BarChart3 size={22} aria-hidden="true" />
        </div>

        <div className="metrics-row">
          <div>
            <span>F1</span>
            <strong>0.577</strong>
            <small>Balance between precision and recall.</small>
          </div>
          <div>
            <span>Precision</span>
            <strong>0.533</strong>
            <small>When the model predicts All-Star, how often it is right.</small>
          </div>
          <div>
            <span>Recall</span>
            <strong>0.629</strong>
            <small>How many future All-Stars the model catches.</small>
          </div>
        </div>
      </section>
    </main>
  )
}

export default App
