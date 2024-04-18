import React, { useEffect, useState } from 'react';
import './SessionSummary.css';
import SummaryGraph from './SummaryGraph';
import { useParams } from 'react-router-dom';
import { Link } from 'react-router-dom';

function SessionSummary() {
    const {sessionIDFromURL} = useParams();
    const [DetectionData, setDetectionData] = useState([]);
    const [sessionId, setSessionId] = useState(null);
    const [startTime, setStartTime] = useState(null);
    const [FlowData, setFlowData] = useState([]);

    // useEffect(() => {
    //     fetch('http://127.0.0.1:8000/api/current_session')
    //         .then(response => response.json())
    //         .then(data => {
    //             setSessionId(data.session_id);
    //             setStartTime(new Date(data.created_at));
    //         });
    // }, []);

    useEffect(() => {
        if (!sessionIDFromURL) return;

        fetch(`http://127.0.0.1:8000/api/session/${sessionIDFromURL}`)
            .then(response => response.json())
            .then(data => {
                setSessionId(data.session_id);
                setStartTime(new Date(data.created_at));
            })
            .catch(error => console.error('Error fetching session data:', error));
    }, [sessionId]);

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

    useEffect(() => {
        console.log('fetching flow data');
        fetch('http://127.0.0.1:8000/api/flow_data')
        .then(response => response.json())
            .then(data => {
                setFlowData(data);
            })
            .catch(error => console.error('Error fetching flow state data:', error));
    }, []);

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
            <div className="chart-container">
                <SummaryGraph DetectionData={DetectionData} startTime={startTime} />
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
            {FlowData.length > 0 ? (
                <div >
                <h2>Flow Count: {FlowData[0].flowCount}</h2>
                <h2>Not in Flow Count:  {FlowData[0].notInFlowCount}</h2>
                <h2>Flow Ratio: {FlowData[0].flowNotFlowRatio}</h2>
            </div>
            ) : (
                <p>No flow data available.</p>
            )}
            <div className="stop-fixed-bottom-2">
                <Link to={`/session-history`}>
                    <button className="stop-button-2"></button>
                </Link>
            </div>
        </div>
    )
}

export default SessionSummary;