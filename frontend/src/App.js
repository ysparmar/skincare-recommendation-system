// App.js — Root application with routing
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import LandingPage   from './pages/LandingPage';
import AnalyzePage   from './pages/AnalyzePage';
import ResultsPage   from './pages/ResultsPage';
import Navbar        from './components/Navbar';
import Footer        from './components/Footer';

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <main>
          <Routes>
            <Route path="/"         element={<LandingPage />} />
            <Route path="/analyze"  element={<AnalyzePage />} />
            <Route path="/results"  element={<ResultsPage />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
