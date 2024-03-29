import React from 'react';
import './SessionSummary.css';
import { Link } from 'react-router-dom';

function SessionSummary() {

    return (
        <div className="session-content">
            <h1>Session Summary</h1>
            <div className="productivity-score">
                <p>Productivity Score: </p>
                <br></br>
            </div>       
        </div>
    )
}

export default SessionSummary;