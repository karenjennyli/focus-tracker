import './App.css';
import React from 'react';

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
      <div className="intro">
        <h1>Focus Tracker App</h1>
        <p>This App enables users to measure their focus and associated distractions during work sessions to help them identify actionable steps to improve their productivity.</p>
        <button className="get-started">Get Started</button>
      </div>
    </div>
  );
}

export default App;
