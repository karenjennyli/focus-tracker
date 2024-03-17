import React, { useEffect, useState } from 'react';
import './YawningData.css'; // Assuming your styles are in YawningData.css

function YawningData() {
    const [yawningData, setYawningData] = useState([]);
    // Add state to track the current session ID. This is initialized in the run.py file
    const [sessionId, setSessionId] = useState(null);

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

        const fetchYawningData = () => {
            fetch(`http://127.0.0.1:8000/api/yawning-data/?session_id=${sessionId}`)
                .then(response => response.json())
                .then(data => {
                    console.log('Updating yawning data for session:', sessionId);
                    setYawningData(data); // Update state with data from the current session
                })
                .catch(error => console.error('Error fetching yawning data:', error));
        };

        const intervalId = setInterval(fetchYawningData, 1000); // Poll every 1000 milliseconds (1 second)

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
            {yawningData.length > 0 ? (
                <table className="yawning-table">
                    <thead>
                        <tr>
                            <th>Timestamp</th>
                            <th>Distraction Type</th>
                            <th>Aspect Ratio</th>
                        </tr>
                    </thead>
                    <tbody>
                        {yawningData.slice(0, 6).map((data, index) => (
                            <tr key={index}>
                                <td>{parseAndFormatTime(data.timestamp)}</td>
                                <td>{data.detection_type}</td>
                                <td>{data.aspect_ratio.toFixed(2)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            ) : (
                <p>No yawning data available for session {sessionId}.</p>
            )}
        </div>
    );
}

export default YawningData;


