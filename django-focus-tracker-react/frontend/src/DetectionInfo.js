import React, { useEffect, useState } from 'react';
import './DetectionData.css'; // Assuming your styles are in DetectionData.css

function DetectionData() {
    const [DetectionData, setDetectionData] = useState([]);
    // Add state to track the current session ID. This is initialized in the run.py file
    const [sessionId, setSessionId] = useState(null);
    const baseURL = 'http://127.0.0.1:8000';

    useEffect(() => {
        // Fetch the current session_id from the backend
        fetch('http://127.0.0.1:8000/api/current_session')
            .then(response => response.json())
            .then(data => {
                setSessionId(data.session_id); // Update the sessionId state
                console.log(data);
            });
    }, []); // Empty array means this runs once on component mount

    useEffect(() => {
        if (!sessionId) return; // Don't fetch data if session ID hasn't been set yet

        const fetchDetectionData = () => {
            fetch(`http://127.0.0.1:8000/api/detection-data/?session_id=${sessionId}`)
                .then(response => response.json())
                .then(data => {
                    console.log('Updating detection data for session:', sessionId);
                    setDetectionData(data); // Update state with data from the current session
                })
                .catch(error => console.error('Error fetching distraction data:', error));
        };

        const intervalId = setInterval(fetchDetectionData, 1000); // Poll every 1000 milliseconds (1 second)

        return () => clearInterval(intervalId); // Cleanup interval on unmount
    }, [sessionId]); // Rerun this effect if sessionId changes

    // Helper function to format timestamp
    const parseAndFormatTime = (timestamp) => {
    // Extract the time part (HH:MM:SS) from the timestamp
    const timePart = timestamp.split('T')[1].split('Z')[0];
    let [hours, minutes] = timePart.split(':');

    // Convert hours to number 
    hours = parseInt(hours, 10);

    // Determine AM or PM
    const ampm = hours >= 12 ? 'PM' : 'AM';

    // Convert to 12-hour format
    hours = hours % 12;
    hours = hours ? hours : 12; // the hour '0' should be '12'

    // Return formatted time string
    return `${hours}:${minutes} ${ampm}`;
};

    return (
        <div>
            <h1>Current Session</h1>
            <h2>Real-Time Updates</h2>
            {DetectionData.length > 0 ? (
                <table className="detection-table">
                    <thead>
                        <tr>
                            <th>Timestamp</th>
                            <th>Distraction Type</th>
                            <th>Image</th>
                        </tr>
                    </thead>
                    <tbody>
                        {DetectionData.slice(0, 6).map((data, index) => (
                            <tr key={index}>
                                <td>{parseAndFormatTime(data.timestamp)}</td>
                                <td>{data.detection_type}</td>
                                <td>
                                    {/* Conditionally render image if URL is available */}
                                    {data.image_url && (
                                        <img src={baseURL + data.image_url} alt="Distraction" style={{ width: '125px' }} />
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            ) : (
                <p>No distraction data available for session {sessionId}.</p>
            )}
        </div>
    );
}

export default DetectionData;


