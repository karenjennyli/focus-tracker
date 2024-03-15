import React, { useEffect, useState } from 'react';

function YawningData() {
    const [yawningData, setYawningData] = useState([]);

    useEffect(() => {
        fetch('http://127.0.0.1:8000/api/yawning-data/')
            .then(response => response.json())
            .then(data => setYawningData(data))
            .catch(error => console.error('Error fetching yawning data:', error));
    }, []);

    return (
        <div>
            <h2>Yawning Detection Data</h2>
            {yawningData.map((data, index) => (
                <div key={index}>
                    <p>User ID: {data.user_id}</p>
                    <p>Event Type: {data.detection_type}</p>
                    <p>Timestamp: {data.timestamp}</p>
                    <p>Aspect Ratio: {data.aspect_ratio}</p>
                </div>
            ))}
        </div>
    );
}

export default YawningData;
