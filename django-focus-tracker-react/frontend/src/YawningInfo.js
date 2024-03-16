import React, { useEffect, useState } from 'react';

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

    return (
        <div>
            <h2>Yawning Detection Data</h2>
            {yawningData.length > 0 ? (
                yawningData.map((data, index) => (
                    <div key={index}>
                        <p>User ID: {data.user_id}</p>
                        <p>Event Type: {data.detection_type}</p>
                        <p>Timestamp: {data.timestamp}</p>
                        <p>Aspect Ratio: {data.aspect_ratio}</p>
                    </div>
                ))
            ) : (
                <p>No yawning data available for session {sessionId}.</p>
            )}
        </div>
    );
}

export default YawningData;


