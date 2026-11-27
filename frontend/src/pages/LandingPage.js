// pages/LandingPage.js — Professional landing page
import React from 'react';
import { Link } from 'react-router-dom';
import './LandingPage.css';

const FEATURES = [
  {
    icon: '🤳',
    title: 'Facial Image Analysis',
    desc: 'Upload your photo and our EfficientNet-B0 model analyzes acne severity with confidence scores.',
  },
  {
    icon: '📋',
    title: 'Smart Questionnaire',
    desc: 'Answer a few questions about your skin type, concerns, and preferences to refine your profile.',
  },
  {
    icon: '🎨',
    title: 'Skin Tone Estimation',
    desc: 'Optional K-Means based skin tone analysis using your facial image for additional context.',
  },
  {
    icon: '🧴',
    title: 'Personalized Recommendations',
    desc: 'Receive ranked Korean skincare product recommendations powered by TF-IDF + Cosine Similarity.',
  },
  {
    icon: '💊',
    title: 'Treatment Guidance',
    desc: 'Discover active ingredients and treatments matched to your specific skin concerns.',
  },
  {
    icon: '🔍',
    title: 'Ingredient Transparency',
    desc: 'See exactly why each product is recommended — ingredients, scores, and concern matches.',
  },
];

const STEPS = [
  { num: '01', label: 'Upload Photo',      desc: 'Take or upload a clear frontal facial photo.' },
  { num: '02', label: 'Fill Questionnaire',desc: 'Tell us about your skin type and concerns.'   },
  { num: '03', label: 'Get Your Profile',  desc: 'Review your unified AI-powered skin profile.' },
  { num: '04', label: 'See Recommendations',desc:'Browse personalized products with explanations.'},
];

export default function LandingPage() {
  return (
    <div className="landing">

      {/* ── Hero ──────────────────────────────────────────── */}
      <section className="hero">
        <div className="container hero-inner">
          <div className="hero-text">
            <div className="hero-badge">🌿 ML-Powered Skincare</div>
            <h1>Discover Your <br /><span className="gradient-text">Perfect Skincare</span><br />Routine</h1>
            <p>
              A hybrid AI system that combines facial image analysis, smart questionnaires,
              and machine learning to recommend the most suitable Korean skincare products for your unique skin.
            </p>
            <div className="hero-actions">
              <Link to="/analyze" className="btn btn-primary btn-lg">
                Analyze My Skin ✨
              </Link>
              <a href="#how-it-works" className="btn btn-outline btn-lg">
                How It Works
              </a>
            </div>
            <div className="hero-stats">
              <div className="stat">
                <span className="stat-value">1,457</span>
                <span className="stat-label">Training Images</span>
              </div>
              <div className="stat-divider" />
              <div className="stat">
                <span className="stat-value">4</span>
                <span className="stat-label">Severity Levels</span>
              </div>
              <div className="stat-divider" />
              <div className="stat">
                <span className="stat-value">K-Beauty</span>
                <span className="stat-label">Products Database</span>
              </div>
            </div>
          </div>
          <div className="hero-visual">
            <div className="hero-card-mockup">
              <div className="mockup-header">
                <span className="mockup-dot red" />
                <span className="mockup-dot yellow" />
                <span className="mockup-dot green" />
                <span className="mockup-title">Skin Analysis</span>
              </div>
              <div className="mockup-body">
                <div className="mockup-face">🙂</div>
                <div className="mockup-result">
                  <div className="mockup-label">Acne Severity</div>
                  <div className="mockup-value">Moderate</div>
                  <div className="mockup-bar">
                    <div className="mockup-fill" style={{ width: '65%', background: '#FF9800' }} />
                  </div>
                  <div className="mockup-confidence">Confidence: 82%</div>
                </div>
                <div className="mockup-tags">
                  <span>Oily Skin</span>
                  <span>Acne-Prone</span>
                  <span>Sensitive</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Features ──────────────────────────────────────── */}
      <section className="section features-section">
        <div className="container">
          <div className="section-header">
            <h2>Everything Your Skin Needs</h2>
            <p>Combining deep learning, content-based filtering, and K-beauty expertise.</p>
          </div>
          <div className="features-grid">
            {FEATURES.map((f, i) => (
              <div className="feature-card card" key={i}>
                <div className="feature-icon">{f.icon}</div>
                <h3>{f.title}</h3>
                <p>{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── How It Works ──────────────────────────────────── */}
      <section className="section how-it-works" id="how-it-works">
        <div className="container">
          <div className="section-header">
            <h2>How It Works</h2>
            <p>Four simple steps to your personalized skincare routine.</p>
          </div>
          <div className="steps-grid">
            {STEPS.map((s, i) => (
              <div className="step-card" key={i}>
                <div className="step-num">{s.num}</div>
                <h3>{s.label}</h3>
                <p>{s.desc}</p>
                {i < STEPS.length - 1 && <div className="step-arrow">→</div>}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ───────────────────────────────────────────── */}
      <section className="section cta-section">
        <div className="container cta-inner">
          <h2>Ready to Know Your Skin Better?</h2>
          <p>It takes less than 2 minutes. No signup required.</p>
          <Link to="/analyze" className="btn btn-primary btn-lg">
            Start Free Analysis →
          </Link>
        </div>
      </section>

    </div>
  );
}
