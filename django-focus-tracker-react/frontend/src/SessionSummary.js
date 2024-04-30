import React, { useEffect, useState } from 'react';
import { HStack, Heading, VStack, Card, Flex, Text } from '@chakra-ui/react';
import './SessionSummary.css';
import { Chart, registerables } from 'chart.js';
import DetectionsBarChart from './DetectionsBarChart';
import EventList from './EventList';
import FlowFocusPieChart from './FlowFocusPieChart';
import DetectionsScatterPlot from './DetectionsScatterPlot';
import { useParams } from 'react-router-dom';
import { Link } from 'react-router-dom';
import ToggleFlowFocusButton from './ToggleFlowFocusButton';
import axios from 'axios';
Chart.register(...registerables);

function SessionSummary() {
    const {sessionIDFromURL} = useParams();
    const [selectedButton, setSelectedButton] = useState('Flow');
    const [DetectionData, setDetectionData] = useState([]);
    const [lastDetectionData, setLastDetectionData] = useState([]);
    const [sessionId, setSessionId] = useState(null);
    const [startTime, setStartTime] = useState(null);
    const [FlowData, setFlowData] = useState([]);
    const [FocusData, setFocusData] = useState([]);
    const [sessionLength, setSessionLength] = useState(0);
    const [flowTime, setFlowTime] = useState(0);
    const [focusTime, setFocusTime] = useState(0);

    // create and set current session id
    useEffect(() => {
        axios.post('http://127.0.0.1:8000/api/current_session', {
            session_id: 'none',
        })
        .then(response => {
            console.log(response.data);
        })
        .catch(error => {
            console.error('Error:', error);
        });
    
    }, []); // Empty array means this runs once on component mount

    useEffect(() => {
        if (!sessionIDFromURL) return;

        fetch(`http://127.0.0.1:8000/api/session/${sessionIDFromURL}`)
            .then(response => response.json())
            .then(data => {
                setSessionId(data.session_id);
                setStartTime(new Date(data.created_at));
            })
            .catch(error => console.error('Error fetching session data:', error));
    }, [sessionId, sessionIDFromURL]);

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
        const fetchData = async () => {
            try {
                const response = await fetch(`http://127.0.0.1:8000/api/session_length/?session_id=${sessionId}`);
                const data = await response.json();
                setSessionLength(data[0].session_length);
                const session_length = data[0].session_length;
    
                const flowResponse = await fetch(`http://127.0.0.1:8000/api/flow_data/?session_id=${sessionId}`);
                const flowData = await flowResponse.json();
                setFlowData(flowData);
                setFlowTime(Math.round(flowData[0].flowCount / (flowData[0].flowCount + flowData[0].notInFlowCount) * session_length));
    
                const focusResponse = await fetch(`http://127.0.0.1:8000/api/focus_data/?session_id=${sessionId}`);
                const focusData = await focusResponse.json();
                setFocusData(focusData);
                setFocusTime(Math.round(focusData[0].focusCount / (focusData[0].focusCount + focusData[0].notInFocusCount) * session_length));
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        };
    
        fetchData();
    }, [sessionId]);

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

    // Format timer to HH:MM:SS
    const formatTime = (seconds) => {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds - (hours * 3600)) / 60);
        const sec = seconds - (hours * 3600) - (minutes * 60);

        return [hours, minutes, sec];
    };

    return (
        <VStack spacing={3}>
            <Heading as="h1" fontSize="5xl" fontWeight="bold" color="white" mt={6} mb={2}>
              Session Summary
            </Heading>
            <HStack spacing={8}>
                <VStack spacing={3}>
                    <HStack>
                        <Card h='200px' w='270px'>
                            <Flex direction='column' align='center' justify='center' h='100%'>
                                <VStack align='center' justify='center' spacing={0}>
                                    <Text fontSize='md' color='gray.500'>
                                        Total Time
                                    </Text>
                                    <Text fontFamily="monospace" fontSize="4xl" marginBottom={3}>
                                        {formatTime(sessionLength).map(time => time.toString().padStart(2, '0')).join(':')}
                                    </Text>
                                    <HStack spacing={6}>
                                        <VStack>
                                            <Text fontSize='md' color='gray.500'>
                                                Flow
                                            </Text>
                                            <Text fontFamily="monospace" fontSize="xl">
                                                {formatTime(flowTime).map(time => time.toString().padStart(2, '0')).join(':')}
                                            </Text>
                                        </VStack>
                                        <VStack>
                                        <Text fontSize='md' color='gray.500'>
                                            Focus
                                        </Text>
                                        <Text fontFamily="monospace" fontSize="xl">
                                            {formatTime(focusTime).map(time => time.toString().padStart(2, '0')).join(':')}
                                        </Text>
                                    </VStack>
                                    </HStack>
                                </VStack>
                            </Flex>
                        </Card>
                        {FlowData.length > 0 ? (
                            <div style={{ width: '220px', height: '220px' }}>
                                <FlowFocusPieChart FlowData={FlowData} FocusData={FocusData} flowFocus={'Flow'} />
                            </div>
                        ) : (
                            <p>No flow data available.</p>
                        )    
                        }
                        {FocusData.length > 0 ? (
                            <div style={{ width: '220px', height: '220px' }}>
                                <FlowFocusPieChart FlowData={FlowData} FocusData={FocusData} flowFocus={'Focus'} />
                            </div>
                        ) : (
                            <p>No flow data available.</p>
                        )    
                        }
                    </HStack>
                    {(DetectionData.length > 0 || FlowData.length > 0 || FocusData.length > 0 ) ? (
                        <DetectionsScatterPlot DetectionData={DetectionData} ProcessedFlowData={FlowData} ProcessedFocusData={FocusData} startTime={startTime} selectedButton={selectedButton}/>
                    ) : (
                        <p>No data to display.</p>
                        // // scatter plot with one point 
                        // <DetectionsScatterPlot DetectionData={[
                        //     { detection_type: 'gaze', timestamp: new Date().toISOString() }
                        // ]}
                        //     ProcessedFlowData={FlowData} startTime={startTime}
                        // />
                    )}
                    <ToggleFlowFocusButton selectedButton={selectedButton} setSelectedButton={setSelectedButton} />
                </VStack>
                <VStack spacing={3}>
                    <Heading as='h1' fontSize='2xl' fontWeight='bold' color='white' mt={0} mb={4}>
                        Detected Events
                    </Heading>
                    <DetectionsBarChart lastDetectionData={lastDetectionData} displayTitle={false} />
                    <EventList detectionData={DetectionData} />
                </VStack>
            </HStack>
        </VStack>
    )
}

export default SessionSummary;