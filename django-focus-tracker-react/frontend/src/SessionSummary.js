import React, { useEffect, useState } from 'react';
import './SessionSummary.css';
// import { Link } from 'react-router-dom';

function SessionSummary() {
    const [DetectionData, setDetectionData] = useState([]);
    const [sessionId, setSessionId] = useState(null);

    useEffect(() => {
        fetch('http://127.0.0.1:8000/api/current_session')
            .then(response => response.json())
            .then(data => {
                setSessionId(data.session_id);
            });
    }, []);

    useEffect(() => {
        if (!sessionId) return;

        fetch(`http://127.0.0.1:8000/api/detection-data/?session_id=${sessionId}`)
            .then(response => response.json())
            .then(data => {
                const filteredData = filterLatestEntries(data);
                setDetectionData(filteredData);
            })
            .catch(error => console.error('Error fetching detection data:', error));
    }, [sessionId]);

    // Filter the data to keep only the most recent entry for each detection type
    const filterLatestEntries = (data) => {
        const latestEntriesMap = {};

        data.forEach(item => {
            // Normalize "gaze left" and "gaze right" to "gaze"
            const detectionType = item.detection_type.includes('gaze') ? 'gaze' : item.detection_type;

            if (!latestEntriesMap[detectionType] || new Date(item.timestamp) > new Date(latestEntriesMap[detectionType].timestamp)) {
                latestEntriesMap[detectionType] = {
                  ...item,
                  detection_type: detectionType // Ensure the detection type is correctly set for the "gaze" group
                };
            }
        });


        return Object.values(latestEntriesMap);
    };

    return (
        <div className="session-content">
            <h1>Session Summary</h1>
            <div className="productivity-score">
                <p>Productivity Score: </p>
                <br></br>
            </div>
            <h2>Distracted Behaviors Detected</h2>
            {DetectionData.length > 0 ? (
                <table className="detection-table-summary">
                    <thead>
                        <tr>
                            <th>Distraction</th>
                            <th>Frequency</th>
                        </tr>
                    </thead>
                    <tbody>
                        {DetectionData.map((data, index) => (
                            <tr key={index}>
                                <td>{data.detection_type}</td>
                                <td>{data.frequency}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            ) : (
                <p>No distraction data available for session {sessionId}.</p>
            )}       
        </div>
    )
}

export default SessionSummary;