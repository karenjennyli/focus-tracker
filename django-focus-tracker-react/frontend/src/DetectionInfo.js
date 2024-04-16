import React, { useEffect, useState } from 'react';
import './DetectionData.css';
import { Link } from 'react-router-dom';
import LiveGraph from './LiveGraph';
import Webcam from "react-webcam";
import EventList from './EventList';
import WebcamCapture from './WebcamCapture';

import {
    VStack,
    HStack,
    Text,
    Heading,
} from '@chakra-ui/react';

function DetectionData() {
    const [DetectionData, setDetectionData] = useState([]);
    const [ProcessedFlowData, setProcessedFlowData] = useState([]);
    const [Events, setEvents] = useState([]);
    // Add state to track the current session ID. This is initialized in the run.py file
    const [sessionId, setSessionId] = useState(null);

    const videoConstraints = {
        width: 1920,
        height: 1080,
        facingMode: "user",
        deviceId: "e4fc9040a6bbd234b0da54ee7c9e5e1796b5ced07398f1ae418c9139b56beb69"
    }

    // get current session id
    useEffect(() => {
        // Fetch the current session_id from the backend
        fetch('http://127.0.0.1:8000/api/current_session')
            .then(response => response.json())
            .then(data => {
                setSessionId(data.session_id); // Update the sessionId state
                console.log(data);
            });
    }, []); // Empty array means this runs once on component mount

    // fetch detection data
    useEffect(() => {
        if (!sessionId) return; // Don't fetch data if session ID hasn't been set yet

        const fetchData = () => {
            fetch(`http://127.0.0.1:8000/api/detection-data/?session_id=${sessionId}`)
                .then(response => response.json())
                .then(data => {
                    console.log('Updating detection data for session:', sessionId);
                    setDetectionData(data); // Update state with data from the current session
                    // set the events state with the latest detection data
                    setEvents(data.map((event) => ({
                        timestamp: parseAndFormatTime(event.timestamp),
                        eventType: event.detection_type,
                        imageUrl: event.image_url,
                        info: event.aspect_ratio,
                    })));
                })
                .catch(error => console.error('Error fetching distraction data:', error));
            fetch(`http://127.0.0.1:8000/api/flow_data`)
                .then(response => response.json())
                .then(data => {
                    console.log('Updating flow data:');
                    setProcessedFlowData(data);
                })
                .catch(error => console.error('Error fetching flow data:', error));
        };

        const intervalId = setInterval(fetchData, 500); // Poll every 500 milliseconds (0.5 second)

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
        <VStack spacing={8}>
            <Heading as="h1" fontSize="5xl" fontWeight="bold" color="white" mt={6} mb={2}>
              Current Session
            </Heading>
            <HStack spacing={8}>
                <VStack spacing={8}>
                    <HStack spacing={8}>
                    <div className="timer-display">
                        Session Length: {formatTime(timer)}
                    </div>
                    <Webcam className='webcam' audio={false} mirrored={true} videoConstraints={videoConstraints}/>
                    </HStack>
                    <div className='graph-container'>
                        <LiveGraph DetectionData={DetectionData} ProcessedFlowData={ProcessedFlowData}/>
                    </div>
                </VStack>
                <VStack spacing={2} justify='start' h='full'>
                    <Heading as="h1" fontSize="3xl" fontWeight="bold" color="white">
                        Latest Events
                    </Heading>
                    {<EventList events={Events} />}
                </VStack>
            </HStack>
            <div className="stop-fixed-bottom">
                    <Link to="/session-summary">
                        <button className="stop-button"></button>
                    </Link>
            </div>
        </VStack>
    );
}

export default DetectionData;
