import './App.css';
import React from 'react';
import Chart from "./Chart";
import PChart from "./PieChart";
import LChart from "./LineChart";

function App() {
  return (
    <div className="App">
      <Header />
      <MainContent />
    </div>
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
        <button className="get-started">Get Started</button>
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
