// import React, { useEffect, useState } from 'react';

// function YawningData() {
//     const [yawningData, setYawningData] = useState([]);

//     useEffect(() => {
//         fetch('http://127.0.0.1:8000/api/yawning-data/')
//             .then(response => response.json())
//             .then(data => setYawningData(data))
//             .catch(error => console.error('Error fetching yawning data:', error));
//     }, []);

//     return (
//         <div>
//             <h2>Yawning Detection Data</h2>
//             {yawningData.map((data, index) => (
//                 <div key={index}>
//                     <p>User ID: {data.user_id}</p>
//                     <p>Event Type: {data.detection_type}</p>
//                     <p>Timestamp: {data.timestamp}</p>
//                     <p>Aspect Ratio: {data.aspect_ratio}</p>
//                 </div>
//             ))}
//         </div>
//     );
// }

// export default YawningData;

// import React, { useEffect, useState } from 'react';

// function YawningData() {
//     const [yawningData, setYawningData] = useState([]);

//     useEffect(() => {
//         const fetchYawningData = () => {
//             fetch('http://127.0.0.1:8000/api/yawning-data/')
//                 .then(response => response.json())
//                 .then(data => {
//                     console.log('Updating yawning data');
//                     setYawningData(data);
//                 })
//                 .catch(error => console.error('Error fetching yawning data:', error));
//         };

//         // Start polling for yawning data immediately and repeat every 5000 milliseconds (5 seconds)
//         const intervalId = setInterval(fetchYawningData, 1000);

//         // Clear the interval when the component unmounts
//         return () => clearInterval(intervalId);
//     }, []); // The empty array ensures this effect runs only once when the component mounts

//     return (
//         <div>
//             <h2>Yawning Detection Data</h2>
//             {yawningData.length > 0 ? (
//                 yawningData.map((data, index) => (
//                     <div key={index}>
//                         <p>User ID: {data.user_id}</p>
//                         <p>Event Type: {data.detection_type}</p>
//                         <p>Timestamp: {data.timestamp}</p>
//                         <p>Aspect Ratio: {data.aspect_ratio}</p>
//                     </div>
//                 ))
//             ) : (
//                 <p>No yawning data available.</p>
//             )}
//         </div>
//     );
// }

// export default YawningData;

import React, { useEffect, useState } from 'react';

function YawningData() {
    const [yawningData, setYawningData] = useState([]);
    const [sessionId, setSessionId] = useState(null); // Add state to track the current session ID

    // useEffect(() => {
    //     // Example way to generate a new session ID - in a real application, this might come from your backend
    //     const newSessionId = `session_${new Date().getTime()}`;
    //     setSessionId(newSessionId);
    //     setYawningData([]);
    // }, []); // This effect runs once on component mount to simulate starting a new session

     // Method to manually start a new session for demonstration
     const startNewSession = () => {
        const newSessionId = `session_${new Date().getTime()}`; // Generate a new session ID
        setSessionId(newSessionId); // Update sessionId state
        setYawningData([]); // Clear previous session's data
    };

    useEffect(() => {
        if (!sessionId) return; // Don't fetch data if session ID hasn't been set yet

        const fetchYawningData = () => {
            fetch(`http://127.0.0.1:8000/api/yawning-data/?session_id=${sessionId}`) // Assuming your API can filter by session ID
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
            <button onClick={startNewSession}>Start New Session</button> {/* Button to start a new session */}
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


