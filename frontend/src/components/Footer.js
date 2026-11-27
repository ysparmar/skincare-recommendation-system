// components/Footer.js
import React from 'react';
import './Footer.css';

export default function Footer() {
  return (
    <footer className="footer">
      <div className="container footer-inner">
        <div className="footer-brand">
          <span>✨ SkinSense<strong>AI</strong></span>
          <p>Personalized skincare powered by Machine Learning.</p>
        </div>
        <div className="footer-disclaimer">
          <p>
            ⚠️ <strong>Disclaimer:</strong> This application is for educational and informational purposes only.
            It does not provide medical advice or replace professional dermatological consultation.
          </p>
        </div>
        <div className="footer-credit">
          <p>BTech CSE — Machine Learning Project · 2026</p>
        </div>
      </div>
    </footer>
  );
}
