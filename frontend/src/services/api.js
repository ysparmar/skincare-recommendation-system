// services/api.js — Axios API service layer
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000, // 60s for model inference
});

// ─── Health ────────────────────────────────────────────────────────────────
export const checkHealth = () => api.get('/health');

// ─── Image Analysis ────────────────────────────────────────────────────────
export const analyzeImage = (imageFile, estimateTone = true) => {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('estimate_tone', estimateTone.toString());
  return api.post('/analyze-image', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

// ─── Skin Profile ──────────────────────────────────────────────────────────
export const buildProfile = (profileData) =>
  api.post('/build-profile', profileData);

// ─── Recommendations ───────────────────────────────────────────────────────
export const getRecommendations = (skinProfile, options = {}) =>
  api.post('/recommendations', {
    skin_profile:    skinProfile,
    max_results:     options.maxResults     || 10,
    category_filter: options.categoryFilter || null,
    sort_by:         options.sortBy         || 'score',
  });

export const getSimilarProducts = (productName, n = 4) =>
  api.post('/similar-products', { product_name: productName, n });

// ─── Model Insights ────────────────────────────────────────────────────────
export const getModelInfo = () => api.get('/model-info');

// ─── Treatments ────────────────────────────────────────────────────────────
export const getTreatments = (skinType, concerns, sensitivity, ageGroup) =>
  api.post('/treatments', {
    skin_type:   skinType,
    concerns:    concerns,
    sensitivity: sensitivity,
    age_group:   ageGroup,
  });

export default api;
