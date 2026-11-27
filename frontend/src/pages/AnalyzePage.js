// pages/AnalyzePage.js — Multi-step analysis workflow with file upload + webcam capture
import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { analyzeImage, buildProfile, getRecommendations, getTreatments } from '../services/api';
import './AnalyzePage.css';

const STEPS = ['Upload Photo', 'Skin Questionnaire', 'Analyzing...'];

const CONCERNS_OPTIONS = [
  { value: 'acne',           label: 'Acne / Breakouts', emoji: '🔴' },
  { value: 'dryness',        label: 'Dryness',           emoji: '💧' },
  { value: 'pigmentation',   label: 'Pigmentation',      emoji: '🌑' },
  { value: 'redness',        label: 'Redness',           emoji: '🌹' },
  { value: 'dark_spots',     label: 'Dark Spots',        emoji: '⚫' },
  { value: 'open_pores',     label: 'Open Pores',        emoji: '🕳️' },
  { value: 'dull_skin',      label: 'Dull Skin',         emoji: '🌫️' },
  { value: 'anti_aging',     label: 'Anti-Aging',        emoji: '⏳' },
  { value: 'uneven_texture', label: 'Uneven Texture',    emoji: '〰️' },
];

const PRODUCT_PREFS = ['Serum', 'Moisturizer', 'Cleanser', 'Toner', 'Sunscreen', 'Essence', 'Mask', 'Eye Cream'];
const ALLERGEN_OPTIONS = ['Fragrance', 'Alcohol', 'Parabens', 'Sulfates', 'Essential Oils', 'Silicones'];

export default function AnalyzePage() {
  const navigate  = useNavigate();
  const fileRef   = useRef(null);
  const videoRef  = useRef(null);
  const streamRef = useRef(null);

  const [step,          setStep]          = useState(0);
  const [imageFile,     setImageFile]     = useState(null);
  const [imagePreview,  setImagePreview]  = useState(null);
  const [cameraOpen,    setCameraOpen]    = useState(false);
  const [estimateTone,  setEstimateTone]  = useState(true);
  const [imageAnalysis, setImageAnalysis] = useState(null);
  const [loading,       setLoading]       = useState(false);
  const [error,         setError]         = useState('');

  // Questionnaire state (no budget/pricing)
  const [skinType,     setSkinType]     = useState('');
  const [sensitivity,  setSensitivity]  = useState('normal');
  const [concerns,     setConcerns]     = useState([]);
  const [allergies,    setAllergies]    = useState([]);
  const [productPrefs, setProductPrefs] = useState([]);
  const [ageGroup,     setAgeGroup]     = useState('adult');
  const [skipImage,    setSkipImage]    = useState(false);

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    setCameraOpen(false);
  };

  useEffect(() => () => stopCamera(), []);

  const startCamera = async () => {
    setError('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
      streamRef.current = stream;
      setCameraOpen(true);
      setTimeout(() => {
        if (videoRef.current) videoRef.current.srcObject = stream;
      }, 100);
    } catch {
      setError('Unable to access webcam. Please check browser permissions or upload an image file.');
    }
  };

  const capturePhoto = () => {
    if (!videoRef.current) return;
    const canvas = document.createElement('canvas');
    canvas.width  = videoRef.current.videoWidth  || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    canvas.getContext('2d').drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const file = new File([blob], 'webcam_capture.jpg', { type: 'image/jpeg' });
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
      setSkipImage(false);
      stopCamera();
    }, 'image/jpeg', 0.92);
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    setError('');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
      setError('');
    }
  };

  const toggleChip = (value, setter, arr) => {
    setter(arr.includes(value) ? arr.filter(v => v !== value) : [...arr, value]);
  };

  const handleImageAnalysis = async () => {
    if (!imageFile && !skipImage) {
      setError('Please upload or capture a facial photo, or choose to skip image analysis.');
      return;
    }
    stopCamera();
    setLoading(true);
    setError('');
    try {
      if (imageFile && !skipImage) {
        const res = await analyzeImage(imageFile, estimateTone);
        setImageAnalysis(res.data);
      }
      setStep(1);
    } catch (err) {
      const msg = err.response?.data?.message || 'Image analysis failed. Please try again.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitQuestionnaire = async () => {
    if (!skinType) {
      setError('Please select your skin type.');
      return;
    }
    setLoading(true);
    setError('');
    setStep(2);

    try {
      const profilePayload = {
        skin_type:           skinType,
        concerns,
        sensitivity,
        allergies:           allergies.map(a => a.toLowerCase()),
        product_preferences: productPrefs.map(p => p.toLowerCase()),
        age_group:           ageGroup,
        image_analysis:      imageAnalysis?.acne_analysis || null,
      };

      const profileRes  = await buildProfile(profilePayload);
      const skinProfile = profileRes.data.skin_profile;

      const recsRes  = await getRecommendations(skinProfile, { maxResults: 10 });
      const treatRes = await getTreatments(skinType, concerns, sensitivity, ageGroup);

      navigate('/results', {
        state: {
          skinProfile,
          imageAnalysis:   imageAnalysis || null,
          recommendations: recsRes.data.recommendations,
          routine:         recsRes.data.routine || { am: [], pm: [] },
          treatments:      treatRes.data.treatments,
        },
      });
    } catch (err) {
      const msg = err.response?.data?.message || 'An error occurred. Please try again.';
      setError(msg);
      setStep(1);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="analyze-page">
      <div className="container">
        <div className="analyze-progress">
          {STEPS.map((label, i) => (
            <div key={i} className={`progress-step ${i <= step ? 'active' : ''} ${i < step ? 'done' : ''}`}>
              <div className="progress-step-dot">{i < step ? '✓' : i + 1}</div>
              <span>{label}</span>
              {i < STEPS.length - 1 && <div className="progress-connector" />}
            </div>
          ))}
        </div>

        {error && (
          <div className="alert alert-error" style={{ marginBottom: 24 }}>
            ⚠️ {error}
          </div>
        )}

        {step === 0 && (
          <div className="step-panel card">
            <h2>Upload Your Facial Photo</h2>
            <p className="step-desc">
              Upload or capture a clear, well-lit frontal photo for acne severity analysis.
              You can also skip this step and rely on the questionnaire.
            </p>

            {!skipImage && cameraOpen && (
              <div className="camera-box">
                <video ref={videoRef} autoPlay playsInline className="camera-video" />
                <div className="camera-actions">
                  <button type="button" className="btn btn-primary" onClick={capturePhoto}>📸 Capture Photo</button>
                  <button type="button" className="btn btn-outline" onClick={stopCamera}>Cancel</button>
                </div>
              </div>
            )}

            {!skipImage && !cameraOpen && (
              <>
                <div
                  className={`upload-zone ${imagePreview ? 'has-image' : ''}`}
                  onDrop={handleDrop}
                  onDragOver={(e) => e.preventDefault()}
                  onClick={() => !imagePreview && fileRef.current?.click()}
                >
                  {imagePreview ? (
                    <div className="image-preview-wrap">
                      <img src={imagePreview} alt="Preview" className="image-preview" />
                      <button className="remove-image" onClick={(e) => { e.stopPropagation(); setImageFile(null); setImagePreview(null); }}>
                        ✕ Remove
                      </button>
                    </div>
                  ) : (
                    <div className="upload-placeholder">
                      <div className="upload-icon">📷</div>
                      <p>Drag & drop your photo here</p>
                      <span>or click to browse</span>
                      <span className="upload-hint">JPG, PNG, WEBP · Max 16 MB</span>
                    </div>
                  )}
                  <input
                    ref={fileRef}
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    style={{ display: 'none' }}
                    onChange={handleFileChange}
                  />
                </div>

                {!imagePreview && (
                  <div style={{ textAlign: 'center', marginBottom: 16 }}>
                    <button type="button" className="btn btn-outline" onClick={startCamera}>
                      🎥 Use Webcam Instead
                    </button>
                  </div>
                )}
              </>
            )}

            {imageAnalysis?.acne_analysis && (
              <div className="alert alert-success" style={{ marginTop: 16 }}>
                ✅ Image analyzed: Acne severity detected as <strong>{imageAnalysis.acne_analysis.severity}</strong> ({imageAnalysis.acne_analysis.confidence_pct}% confidence)
              </div>
            )}

            <div className="form-group" style={{ marginTop: 20 }}>
              <label className="checkbox-label">
                <input type="checkbox" checked={estimateTone} onChange={e => setEstimateTone(e.target.checked)} />
                Also estimate approximate skin tone from photo
              </label>
            </div>

            <div className="form-group">
              <label className="checkbox-label">
                <input type="checkbox" checked={skipImage} onChange={e => { setSkipImage(e.target.checked); setImageFile(null); setImagePreview(null); stopCamera(); }} />
                Skip photo — use questionnaire only
              </label>
            </div>

            <button
              className="btn btn-primary btn-lg"
              onClick={handleImageAnalysis}
              disabled={loading}
              style={{ marginTop: 8 }}
            >
              {loading ? '⏳ Analyzing...' : (imageFile && !skipImage ? 'Analyze Photo & Continue →' : 'Continue to Questionnaire →')}
            </button>
          </div>
        )}

        {step === 1 && (
          <div className="step-panel card">
            <h2>Tell Us About Your Skin</h2>
            <p className="step-desc">Your answers help personalize your skincare recommendations.</p>

            <div className="form-group">
              <label className="form-label">What is your skin type? *</label>
              <div className="chip-group">
                {['dry', 'oily', 'normal', 'combination', 'sensitive'].map(t => (
                  <button
                    key={t}
                    className={`chip ${skinType === t ? 'selected' : ''}`}
                    onClick={() => setSkinType(t)}
                  >
                    {t.charAt(0).toUpperCase() + t.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">What are your main skin concerns? (Select all that apply)</label>
              <div className="chip-group">
                {CONCERNS_OPTIONS.map(c => (
                  <button
                    key={c.value}
                    className={`chip ${concerns.includes(c.value) ? 'selected' : ''}`}
                    onClick={() => toggleChip(c.value, setConcerns, concerns)}
                  >
                    {c.emoji} {c.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Skin sensitivity</label>
              <select className="form-select" value={sensitivity} onChange={e => setSensitivity(e.target.value)}>
                <option value="sensitive">Sensitive — reacts easily to new products</option>
                <option value="normal">Normal — occasionally reacts</option>
                <option value="resilient">Resilient — rarely reacts</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Known ingredient sensitivities (optional)</label>
              <div className="chip-group">
                {ALLERGEN_OPTIONS.map(a => (
                  <button
                    key={a}
                    className={`chip ${allergies.includes(a) ? 'selected' : ''}`}
                    onClick={() => toggleChip(a, setAllergies, allergies)}
                  >
                    {a}
                  </button>
                ))}
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Product types you're looking for</label>
              <div className="chip-group">
                {PRODUCT_PREFS.map(p => (
                  <button
                    key={p}
                    className={`chip ${productPrefs.includes(p) ? 'selected' : ''}`}
                    onClick={() => toggleChip(p, setProductPrefs, productPrefs)}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Age group</label>
              <select className="form-select" value={ageGroup} onChange={e => setAgeGroup(e.target.value)}>
                <option value="teen">Teen (13–19)</option>
                <option value="adult">Adult (20–49)</option>
                <option value="senior">Senior (50+)</option>
              </select>
            </div>

            <div className="step-actions">
              <button className="btn btn-outline" onClick={() => setStep(0)}>← Back</button>
              <button className="btn btn-primary btn-lg" onClick={handleSubmitQuestionnaire} disabled={loading}>
                {loading ? '⏳ Building Your Profile...' : 'Get My Recommendations ✨'}
              </button>
            </div>
          </div>
        )}

        {step === 2 && (
          <div className="step-panel card loading-panel">
            <div className="loading-overlay">
              <div className="spinner" />
              <h3>Building Your Skin Profile</h3>
              <p>Running the recommendation engine on your profile...</p>
              <div className="loading-steps">
                <div className="loading-step active">✓ Skin profile built</div>
                <div className="loading-step active">⏳ Fetching personalized recommendations...</div>
                <div className="loading-step">⏳ Building AM/PM routine & treatment guide...</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
