import './App.css';
import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import Chart from "./Chart";
import PChart from "./PieChart";
import LChart from "./LineChart";
// import WebcamStream from './WebcamStream';
import DetectionData from './DetectionInfo';
import CalibrationPage from './CalibrationPage';
import SessionSummary from './SessionSummary';

function App() {
  return (
    <Router>
      <div className="App">
        <Header />
        <Routes> {/* Use Routes instead of Switch */}
          <Route path="/" element={<MainContent />} />
          <Route path="/calibration-page" element={<CalibrationPage />} />
          <Route path="/detection-info" element={<DetectionData />} />
          <Route path="/session-summary" element={<SessionSummary />} />
        </Routes>
      </div>
    </Router>
  );
}

function Header() {
  return (
    <header className="App-header">
      <div className="logo">Logo</div>
      <nav>
        <ul>
          <li>Home</li>
          <li>Features</li>
          <li>About</li>
        </ul>
      </nav>
    </header>
  );
}

function MainContent() {
  return (
    <div className="main-content">
      <h1>Focus Tracker App</h1>
      <div className="intro">
        <p>This App enables users to measure their focus and associated distractions during <br></br>work sessions to help them identify actionable steps to improve their productivity.</p>
        <br></br>
        <Link to="/calibration-page"><button className="get-started">Get Started</button></Link>
      </div>
      <div className="charts-container">
        <div className="chart">
          <Chart />
        </div>
        <div className="pchart">
          <PChart />
        </div>
        <div className="lchart">
          <LChart />
        </div>
      </div>
    </div>
  );
}

export default App;
