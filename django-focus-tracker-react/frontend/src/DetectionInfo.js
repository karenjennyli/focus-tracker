import React, { useEffect, useState } from 'react';
import './DetectionData.css';
import { Scatter } from 'react-chartjs-2';
import 'chart.js/auto';
import 'chartjs-adapter-moment';
import { Chart, registerables } from 'chart.js';
Chart.register(...registerables);

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

        const intervalId = setInterval(fetchDetectionData, 500); // Poll every 500 milliseconds (0.5 second)

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

        // Prepare chart data
    const chartData = {
        datasets: [
            {
                label: 'Yawn',
                data: DetectionData.filter(d => d.detection_type === 'yawn')
                    .map(d => ({ x: new Date(d.timestamp), y: 1 })),
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                borderColor: 'rgba(0, 0, 0, 0.8)', // Make lines darker,
                pointRadius: 5,
            },
            {
                label: 'Sleep',
                data: DetectionData.filter(d => d.detection_type === 'sleep')
                    .map(d => ({ x: new Date(d.timestamp), y: 2 })),
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgba(0, 0, 0, 0.8)', // Make lines darker,
                pointRadius: 5
            },
            {
                label: 'Gaze',
                data: DetectionData.filter(d => d.detection_type.includes('gaze'))
                    .map(d => ({ x: new Date(d.timestamp), y: 3 })),
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                borderColor: 'rgba(0, 0, 0, 0.8)', // Make lines darker,
                pointRadius: 5
            }
        ]
    };

    const options = {
        scales: {
            x: {
                grid: {
                    color: 'rgba(0, 0, 0, 0.8)', // Darker grid lines
                    lineWidth: 1, // Thicker grid lines
                },
                type: 'time',
                time: {
                    unit: 'minute',
                    tooltipFormat: 'h:mm',
                    displayFormats: {
                        minute: 'h:mm' // Apply the same format for axis labels
                    },
                    round: 'minute'
                },
                title: {
                    display: true,
                    text: 'Time',
                    color: '#000', // Dark font color
                    font: {
                        size: 16 // Larger font size
                    }
                },
                ticks: {
                    color: '#000', // Dark font color for ticks
                    font: {
                        size: 14 // Larger font size for ticks
                    }
                },
            },
            y: {
                grid: {
                    color: 'rgba(0, 0, 0, 0.8)', // Darker grid lines
                    lineWidth: 1, // Thicker grid lines
                },
                // beginAtZero: true,
                ticks: {
                    // This will only work if you have numeric values for y
                    callback: function (value) {
                        if (value === 1) return 'Yawn';
                        else if (value === 2) return 'Sleep';
                        else if (value === 3) return 'Gaze';
                        return null;
                    },
                    color: '#000', // Dark font color for ticks
                    font: {
                        size: 14 // Larger font size for ticks
                    }
                },
                title: {
                    display: true,
                    text: 'Distraction Type', 
                    color: '#000', // Dark font color
                    font: {
                        size: 16 // Larger font size
                    }
                }
            }
        },
        plugins: {
            legend: {
                labels: {
                    color: '#000', // Dark font color for legend
                    font: {
                        size: 14 // Larger font size for legend
                    }
                },
                position: 'top',
            }
        }
    };

    const [timer, setTimer] = useState(0);

    // Update the timer every second
    useEffect(() => {
        const intervalId = setInterval(() => {
            setTimer(prevTimer => prevTimer + 1);
        }, 1000);

        // Cleanup interval on component unmount
        return () => clearInterval(intervalId);
    }, []);

    // Format timer to HH:MM:SS
    const formatTime = (seconds) => {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds - (hours * 3600)) / 60);
        const sec = seconds - (hours * 3600) - (minutes * 60);

        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
    };
    
    return (
        <div>
            <div>
                <h1>Current Session</h1>
                <div className="timer-display">
                    Session Length: {formatTime(timer)}
                </div>
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
            <div className="chart-container">
                {DetectionData.length > 0 && (
                    <Scatter data={chartData} options={options}/>
                )}
            </div>
        </div>
    );
}

export default DetectionData;


