// components/Navbar.js
import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Navbar.css';

export default function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();

  const isActive = (path) => location.pathname === path;

  return (
    <nav className="navbar">
      <div className="container navbar-inner">
        <Link to="/" className="navbar-brand">
          <span className="brand-icon">✨</span>
          <span className="brand-name">SkinSense<span className="brand-ai">AI</span></span>
        </Link>

        <button
          className={`navbar-toggle ${menuOpen ? 'open' : ''}`}
          onClick={() => setMenuOpen(!menuOpen)}
          aria-label="Toggle menu"
        >
          <span /><span /><span />
        </button>

        <ul className={`navbar-links ${menuOpen ? 'open' : ''}`}>
          <li><Link to="/"        className={isActive('/')        ? 'active' : ''} onClick={() => setMenuOpen(false)}>Home</Link></li>
          <li><Link to="/analyze" className={isActive('/analyze') ? 'active' : ''} onClick={() => setMenuOpen(false)}>Analyze Skin</Link></li>
          <li>
            <Link to="/analyze" className="btn btn-primary navbar-cta" onClick={() => setMenuOpen(false)}>
              Get Started
            </Link>
          </li>
        </ul>
      </div>
    </nav>
  );
}
