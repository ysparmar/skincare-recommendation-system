// pages/ResultsPage.js — Skin profile + recommendations + AM/PM routine + treatments + Model Insights + Print
import React, { useState } from 'react';
import { useLocation, Link } from 'react-router-dom';
import { getSimilarProducts, getModelInfo } from '../services/api';
import './ResultsPage.css';

const SEVERITY_COLORS = {
  'Mild':        { bg: '#E8F5E9', color: '#2E7D32', bar: '#4CAF50' },
  'Moderate':    { bg: '#FFF3E0', color: '#E65100', bar: '#FF9800' },
  'Severe':      { bg: '#FFEBEE', color: '#C62828', bar: '#F44336' },
  'Very Severe': { bg: '#F3E5F5', color: '#6A1B9A', bar: '#9C27B0' },
  'Unknown':     { bg: '#F5F5F5', color: '#757575', bar: '#BDBDBD' },
};

export default function ResultsPage() {
  const { state } = useLocation();
  const [activeTab, setActiveTab]           = useState('recommendations');
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [modelInfo, setModelInfo]           = useState(null);
  const [showInsights, setShowInsights]     = useState(false);

  if (!state) {
    return (
      <div className="results-page">
        <div className="container" style={{ textAlign: 'center', paddingTop: 80 }}>
          <h2>No results found</h2>
          <p style={{ color: 'var(--text-secondary)', marginBottom: 24 }}>
            Please complete the skin analysis first.
          </p>
          <Link to="/analyze" className="btn btn-primary">Start Analysis</Link>
        </div>
      </div>
    );
  }

  const { skinProfile, imageAnalysis, recommendations = [], routine = { am: [], pm: [] }, treatments = [] } = state;
  const acne     = imageAnalysis?.acne_analysis;
  const tone     = imageAnalysis?.skin_tone;
  const severity = acne?.severity || 'Unknown';
  const sevStyle = SEVERITY_COLORS[severity] || SEVERITY_COLORS['Unknown'];

  const categories = ['all', ...new Set(recommendations.map(r => r.category?.toLowerCase()).filter(Boolean))];
  const filteredRecs = categoryFilter === 'all'
    ? recommendations
    : recommendations.filter(r => r.category?.toLowerCase() === categoryFilter);

  const handleOpenInsights = async () => {
    setShowInsights(true);
    if (!modelInfo) {
      try {
        const res = await getModelInfo();
        setModelInfo(res.data);
      } catch {
        setModelInfo({ error: true });
      }
    }
  };

  return (
    <div className="results-page">
      <div className="container">
        <div className="results-header">
          <h1>Your Skin Analysis Results</h1>
          <p>Here is your personalized skin profile, product recommendations, and daily routine.</p>
          <div className="header-actions no-print">
            <Link to="/analyze" className="btn btn-outline">← Start New Analysis</Link>
            <button type="button" className="btn btn-outline" onClick={handleOpenInsights}>📊 Model Insights</button>
            <button type="button" className="btn btn-primary" onClick={() => window.print()}>🖨️ Print / Save Summary</button>
          </div>
        </div>

        {/* ── Model Insights Modal ───────────────────────── */}
        {showInsights && (
          <div className="modal-backdrop no-print" onClick={() => setShowInsights(false)}>
            <div className="card modal-card" onClick={e => e.stopPropagation()}>
              <div className="modal-header">
                <h3>Machine Learning Model Insights</h3>
                <button className="modal-close" onClick={() => setShowInsights(false)}>✕</button>
              </div>
              {!modelInfo ? (
                <p>Loading model metrics...</p>
              ) : modelInfo.error ? (
                <p>Could not load model details right now.</p>
              ) : (
                <div className="insights-grid">
                  <div className="insight-box">
                    <div className="insight-label">Acne Severity Classifier</div>
                    <div className="insight-val">{modelInfo.acne_model?.architecture}</div>
                    <div className="insight-sub">{modelInfo.acne_model?.dataset}</div>
                    <div className="insight-metric">
                      Test Accuracy: <strong>{Math.round((modelInfo.acne_model?.test_acc || 0) * 10000) / 100}%</strong>
                      {modelInfo.acne_model?.tta_acc && (
                        <> · TTA: <strong>{Math.round(modelInfo.acne_model.tta_acc * 10000) / 100}%</strong></>
                      )}
                    </div>
                  </div>
                  <div className="insight-box">
                    <div className="insight-label">Recommendation Engine</div>
                    <div className="insight-val">{modelInfo.recommender?.method}</div>
                    <div className="insight-sub">
                      {modelInfo.recommender?.total_products} products · {modelInfo.recommender?.total_features} TF-IDF features · {modelInfo.recommender?.reviews_used?.toLocaleString()} reviews
                    </div>
                  </div>
                  <div className="insight-box">
                    <div className="insight-label">Treatment Knowledge Base</div>
                    <div className="insight-val">{modelInfo.treatment_mapper?.dataset}</div>
                    <div className="insight-sub">{modelInfo.treatment_mapper?.total_records} unique concern-to-ingredient records</div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── Skin Profile Summary ─────────────────────────── */}
        <div className="profile-grid">
          {acne && (
            <div className="card profile-card acne-card">
              <div className="profile-card-label">Acne Severity</div>
              <div className="severity-badge" style={{ background: sevStyle.bg, color: sevStyle.color }}>
                {severity}
              </div>
              <div className="confidence-bar-wrap">
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${acne.confidence_pct}%`, background: sevStyle.bar }} />
                </div>
                <span className="confidence-label">
                  {acne.confidence_pct}% confidence
                  {acne.low_confidence && ' ⚠️ (low)'}
                </span>
              </div>
              {acne.low_confidence && (
                <div className="alert alert-warning" style={{ marginTop: 10, fontSize: 12 }}>
                  Low confidence — questionnaire responses weighted more heavily.
                </div>
              )}
              <div className="prob-grid">
                {Object.entries(acne.all_probabilities || {}).map(([label, prob]) => (
                  <div key={label} className="prob-item">
                    <span className="prob-label">{label}</span>
                    <div className="progress-bar" style={{ height: 5 }}>
                      <div className="progress-fill" style={{ width: `${Math.round(prob * 100)}%` }} />
                    </div>
                    <span className="prob-pct">{Math.round(prob * 100)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="card profile-card">
            <div className="profile-card-label">Skin Profile</div>
            <div className="profile-item">
              <span className="profile-key">Skin Type</span>
              <span className="profile-val">{skinProfile.skin_type?.charAt(0).toUpperCase() + skinProfile.skin_type?.slice(1)}</span>
            </div>
            <div className="profile-item">
              <span className="profile-key">Sensitivity</span>
              <span className="profile-val">{skinProfile.sensitivity}</span>
            </div>
            <div className="profile-item">
              <span className="profile-key">Age Group</span>
              <span className="profile-val">{skinProfile.age_group}</span>
            </div>
            <div className="concerns-list">
              <span className="profile-key">Concerns</span>
              <div className="chip-group" style={{ marginTop: 8 }}>
                {skinProfile.primary_concerns?.map(c => (
                  <span key={c} className="chip selected" style={{ cursor: 'default', fontSize: 12 }}>
                    {c.replace('_', ' ')}
                  </span>
                ))}
              </div>
            </div>
            <div className="profile-completeness">
              <span className="profile-key">Profile Completeness</span>
              <div className="progress-bar" style={{ marginTop: 6 }}>
                <div className="progress-fill" style={{ width: `${Math.round((skinProfile.profile_completeness || 0) * 100)}%` }} />
              </div>
              <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                {Math.round((skinProfile.profile_completeness || 0) * 100)}%
              </span>
            </div>
          </div>

          {tone?.available && (
            <div className="card profile-card tone-card">
              <div className="profile-card-label">Skin Tone Estimate</div>
              <div className="tone-swatch-wrap">
                <div className="tone-swatch" style={{ background: tone.dominant_hex }} />
                <div>
                  <div className="tone-label">{tone.tone_category}</div>
                  <div className="tone-fitzpatrick">Fitzpatrick {tone.fitzpatrick_estimate}</div>
                  <div className="tone-hex">{tone.dominant_hex}</div>
                </div>
              </div>
              <p className="tone-note">{tone.note}</p>
            </div>
          )}
        </div>

        {/* ── Tabs ─────────────────────────────────────────── */}
        <div className="results-tabs no-print">
          <button className={`tab-btn ${activeTab === 'recommendations' ? 'active' : ''}`} onClick={() => setActiveTab('recommendations')}>
            🧴 Product Recommendations ({recommendations.length})
          </button>
          <button className={`tab-btn ${activeTab === 'routine' ? 'active' : ''}`} onClick={() => setActiveTab('routine')}>
            🌞🌙 AM / PM Routine
          </button>
          <button className={`tab-btn ${activeTab === 'treatments' ? 'active' : ''}`} onClick={() => setActiveTab('treatments')}>
            💊 Treatment Guide
          </button>
        </div>

        {/* ── Recommendations Tab ───────────────────────────── */}
        {activeTab === 'recommendations' && (
          <div className="tab-panel">
            {recommendations.length === 0 ? (
              <div className="alert alert-warning">No products found matching your profile.</div>
            ) : (
              <>
                <div className="category-filter no-print">
                  {categories.map(cat => (
                    <button
                      key={cat}
                      className={`chip ${categoryFilter === cat ? 'selected' : ''}`}
                      onClick={() => setCategoryFilter(cat)}
                    >
                      {cat.charAt(0).toUpperCase() + cat.slice(1)}
                    </button>
                  ))}
                </div>

                <div className="product-grid">
                  {filteredRecs.map((product, i) => (
                    <ProductCard key={i} product={product} skinType={skinProfile.skin_type} />
                  ))}
                </div>
              </>
            )}
          </div>
        )}

        {/* ── AM / PM Routine Tab ───────────────────────────── */}
        {activeTab === 'routine' && (
          <div className="tab-panel">
            <div className="routine-grid">
              <div className="card routine-card">
                <h3>☀️ Morning (AM) Routine</h3>
                <p className="routine-sub">Focus on gentle cleansing, hydration, and sun protection.</p>
                {(routine.am || []).length === 0 ? (
                  <p style={{ color: 'var(--text-secondary)' }}>No AM steps matched.</p>
                ) : (
                  <div className="routine-steps">
                    {routine.am.map((item, i) => (
                      <div key={i} className="routine-step-row">
                        <div className="routine-step-num">{i + 1}</div>
                        <div className="routine-step-body">
                          <div className="routine-step-label">{item.step}</div>
                          <div className="routine-step-prod">{item.product_name} <span className="routine-brand">({item.brand})</span></div>
                          {item.reason && <div className="routine-step-reason">✓ {item.reason}</div>}
                        </div>
                        <div className="routine-step-score">{item.score_pct}%</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="card routine-card">
                <h3>🌙 Night (PM) Routine</h3>
                <p className="routine-sub">Focus on cleansing, targeted actives, and barrier repair.</p>
                {(routine.pm || []).length === 0 ? (
                  <p style={{ color: 'var(--text-secondary)' }}>No PM steps matched.</p>
                ) : (
                  <div className="routine-steps">
                    {routine.pm.map((item, i) => (
                      <div key={i} className="routine-step-row">
                        <div className="routine-step-num">{i + 1}</div>
                        <div className="routine-step-body">
                          <div className="routine-step-label">{item.step}</div>
                          <div className="routine-step-prod">{item.product_name} <span className="routine-brand">({item.brand})</span></div>
                          {item.reason && <div className="routine-step-reason">✓ {item.reason}</div>}
                        </div>
                        <div className="routine-step-score">{item.score_pct}%</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ── Treatments Tab ────────────────────────────────── */}
        {activeTab === 'treatments' && (
          <div className="tab-panel">
            <div className="alert alert-info" style={{ marginBottom: 24 }}>
              ℹ️ <strong>Note:</strong> This information is for educational purposes only.
              It does not replace professional dermatological advice.
            </div>

            {treatments.length === 0 ? (
              <div className="alert alert-warning">No treatment data available for your concerns.</div>
            ) : (
              <div className="treatment-grid">
                {treatments.map((t, i) => (
                  <div key={i} className="card treatment-card">
                    <h3>🎯 {t.concern?.replace(/_/g, ' ').toUpperCase()}</h3>

                    {t.note ? (
                      <p className="treatment-note">{t.note}</p>
                    ) : (
                      <>
                        {t.recommended_ingredients?.length > 0 && (
                          <div className="treatment-section">
                            <div className="treatment-label">Recommended Ingredients</div>
                            <div className="chip-group">
                              {t.recommended_ingredients.map(ing => (
                                <span key={ing} className="chip selected" style={{ cursor: 'default', fontSize: 12 }}>{ing}</span>
                              ))}
                            </div>
                          </div>
                        )}

                        {t.concentrations?.length > 0 && (
                          <div className="treatment-section">
                            <div className="treatment-label">Suggested Concentrations</div>
                            <ul className="treatment-list">
                              {t.concentrations.map((c, ci) => <li key={ci}>{c}</li>)}
                            </ul>
                          </div>
                        )}

                        {t.effects?.length > 0 && (
                          <div className="treatment-section">
                            <div className="treatment-label">Expected Benefits</div>
                            <div className="chip-group">
                              {t.effects.map(e => (
                                <span key={e} className="chip" style={{ cursor: 'default', fontSize: 12 }}>{e}</span>
                              ))}
                            </div>
                          </div>
                        )}

                        {t.usage_notes?.length > 0 && (
                          <div className="alert alert-warning" style={{ marginTop: 12, fontSize: 13 }}>
                            {t.usage_notes.map((n, ni) => <div key={ni}>⚠️ {n}</div>)}
                          </div>
                        )}
                      </>
                    )}

                    <div className="treatment-disclaimer">{t.disclaimer}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Product Card Component ───────────────────────────────────────────────────
function ProductCard({ product, skinType }) {
  const [expanded, setExpanded]         = useState(false);
  const [similar, setSimilar]           = useState(null);
  const [loadingSim, setLoadingSim]     = useState(false);
  const scoreColor = product.score_pct >= 75 ? '#4CAF50' : product.score_pct >= 50 ? '#FF9800' : '#F44336';

  const handleFindSimilar = async () => {
    if (similar) {
      setSimilar(null);
      return;
    }
    setLoadingSim(true);
    try {
      const res = await getSimilarProducts(product.product_name, 3);
      setSimilar(res.data.similar_products || []);
    } catch {
      setSimilar([]);
    } finally {
      setLoadingSim(false);
    }
  };

  return (
    <div className="card product-card">
      <div className="product-card-header">
        <div className="product-rank">#{product.rank}</div>
        <div className="product-info">
          <div className="product-name">{product.product_name}</div>
          <div className="product-brand">{product.brand}</div>
        </div>
        <div className="product-score" style={{ color: scoreColor }}>
          {product.score_pct}%
          <span className="score-hint">match</span>
        </div>
      </div>

      <div className="product-meta">
        <span className="product-category">{product.category}</span>
        {product.review_count > 0 && (
          <span className="product-reviews">💬 {product.review_count} reviews</span>
        )}
        {product.skin_type_mentions > 0 && (
          <span className="product-mentions">✓ {product.skin_type_mentions} {skinType} mentions</span>
        )}
      </div>

      <div className="progress-bar" style={{ margin: '10px 0' }}>
        <div className="progress-fill" style={{ width: `${product.score_pct}%`, background: scoreColor }} />
      </div>

      <div className="recommendation-reasons">
        {product.recommendation_reasons?.slice(0, 2).map((r, i) => (
          <div key={i} className="reason-item">✓ {r}</div>
        ))}
      </div>

      {product.matched_concerns?.length > 0 && (
        <div className="chip-group" style={{ marginTop: 10 }}>
          {product.matched_concerns.map(c => (
            <span key={c} className="chip selected" style={{ cursor: 'default', fontSize: 11, padding: '3px 10px' }}>
              {c.replace(/_/g, ' ')}
            </span>
          ))}
        </div>
      )}

      {product.avoided_ingredients?.length > 0 && (
        <div className="alert alert-warning" style={{ marginTop: 10, fontSize: 12 }}>
          ⚠️ Contains: {product.avoided_ingredients.join(', ')} — may not suit sensitive skin.
        </div>
      )}

      <div className="card-footer-actions no-print">
        <button className="expand-btn" onClick={() => setExpanded(!expanded)}>
          {expanded ? '▲ Less detail' : '▼ More detail'}
        </button>
        <button className="expand-btn" onClick={handleFindSimilar} disabled={loadingSim}>
          {loadingSim ? '⏳ Finding...' : (similar ? '✕ Hide similar' : '🔍 Find similar')}
        </button>
      </div>

      {similar && (
        <div className="similar-box">
          <div className="similar-title">Similar Products (Cosine Similarity)</div>
          {similar.length === 0 ? (
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>No similar items found.</div>
          ) : (
            similar.map((s, si) => (
              <div key={si} className="similar-item">
                <span>{s.product_name} <small>({s.brand})</small></span>
                <strong>{Math.round(s.similarity * 100)}%</strong>
              </div>
            ))
          )}
        </div>
      )}

      {expanded && (
        <div className="product-detail">
          {product.recommendation_reasons?.slice(2).map((r, i) => (
            <div key={i} className="reason-item">✓ {r}</div>
          ))}
          {product.ingredients_snippet && (
            <div className="ingredient-snippet">
              <strong>Description & Actives:</strong> {product.ingredients_snippet}
              {product.ingredients_snippet.length >= 299 && '...'}
            </div>
          )}
          {product.review_snippet && (
            <div className="review-snippet">
              <strong>User review:</strong> "{product.review_snippet}"
            </div>
          )}
        </div>
      )}
    </div>
  );
}
