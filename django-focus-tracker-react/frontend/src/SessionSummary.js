import React, { useEffect, useState } from 'react';
import { HStack, Heading, VStack } from '@chakra-ui/react';
import './SessionSummary.css';
import { Chart, registerables } from 'chart.js';
import DetectionsBarChart from './DetectionsBarChart';
import EventList from './EventList';
import FlowPieChart from './FlowPieChart';
import DetectionsScatterPlot from './DetectionsScatterPlot';
import { useParams } from 'react-router-dom';
import { Link } from 'react-router-dom';
Chart.register(...registerables);

function SessionSummary() {
    const {sessionIDFromURL} = useParams();
    const [DetectionData, setDetectionData] = useState([]);
    const [lastDetectionData, setLastDetectionData] = useState([]);
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
                setDetectionData(data);
                const filteredData = filterLatestEntries(data);
                setLastDetectionData(filteredData);

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
            // Normalize "user returned" and "user not recognized" to "user left"
            let detectionType = item.detection_type;
            if (detectionType.includes('gaze')) {
                detectionType = 'gaze';
            } else if (detectionType.includes('user')) {
                detectionType = 'user left';
            }

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
        <VStack spacing={3}>
            <Heading as="h1" fontSize="5xl" fontWeight="bold" color="white" mt={6} mb={2}>
              Session Summary
            </Heading>
            <HStack spacing={8}>
                <VStack spacing={3}>
                    {FlowData.length > 0 ? (
                        <div style={{ width: '250px', height: '250px' }}>
                            <FlowPieChart FlowData={FlowData} />
                        </div>
                    ) : (
                        <p>No flow data available.</p>
                    )    
                    }
                    {DetectionData.length > 0 ? (
                        <DetectionsScatterPlot DetectionData={DetectionData} ProcessedFlowData={FlowData} startTime={startTime} />
                    ) : (
                        <p>No detection data available.</p>
                    )}
                </VStack>
                <VStack spacing={3}>
                    <Heading as='h1' fontSize='2xl' fontWeight='bold' color='white' mt={0} mb={4}>
                        Detected Events
                    </Heading>
                    <DetectionsBarChart lastDetectionData={lastDetectionData} displayTitle={false} />
                    <EventList detectionData={DetectionData} />
                </VStack>
            </HStack>
            <div className="stop-fixed-bottom-2">
                <Link to={`/session-history`}>
                    <button className="stop-button-2"></button>
                </Link>
            </div>
        </VStack>
    )
}

export default SessionSummary;