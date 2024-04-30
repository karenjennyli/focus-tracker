import React, { useEffect, useState } from 'react';
import { Bar } from 'react-chartjs-2';
import { useNavigate } from 'react-router-dom';
import { Line } from 'react-chartjs-2';

import {
    VStack,
    HStack,
    Text,
    Heading,
    Table,
    Thead,
    Tbody,
    Tr,
    Th,
    Td,
    Link,
    Box
} from '@chakra-ui/react';

function SessionHistory() {
    const [chartData, setChartData] = useState({});
    const [sessions, setSessions] = useState([]); // Added to store session details directly
    const navigate = useNavigate();

    useEffect(() => {
        fetch('http://127.0.0.1:8000/api/session_history')
            .then(response => response.json())
            .then(data => {
                const sessionData = data.map(session => ({
                    sessionId: session.session_id,
                    distractions: session.total_distractions,
                    timestamp: new Date(session.timestamp).toLocaleString('default', {
                        year: 'numeric',
                        month: '2-digit',
                        day: '2-digit',
                        hour: '2-digit',
                        minute: '2-digit'
                    })
                }));
                setSessions(sessionData); // Set sessions for the table
                // Filter data to exclude sessions with zero distractions before setting chart data
                const filteredData = sessionData.filter(session => session.distractions > 0);
                setChartData({
                    labels: filteredData.map(item => item.timestamp),
                    datasets: [
                        {
                            label: 'Total Distractions',
                            data: filteredData.map(item => item.distractions),
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                            borderColor: 'rgba(75, 192, 192, 1)',
                            borderWidth: 3,
                            fill: true,
                            tension: 0.1
                        }
                    ],
                    sessionIds: filteredData.map(item => item.sessionId)
                });
            })
            .catch(error => console.error('Error fetching sessions:', error));
    }, []);

    const options = {
        onClick: (event, elements, chart) => {
            if (elements.length > 0) {
                const elementIndex = elements[0].index;
                const session_id = chartData.sessionIds[elementIndex];
                navigate(`/session-summary/${session_id}`);
            }
        },
        responsive: true,
        scales: {
          y: {
            beginAtZero: true,
            grid: {
                color: 'rgba(255, 255, 255, 0.1)',  // Lighter grid lines
                borderWidth: 1
            },
            ticks: {
              color: '#FFF',
              font: {
                size: 14,
                family: 'Arial', 
                weight: 'bold'
              },
              stepSize: 1,
            }
          },
          x: {
            ticks: {
              color: '#FFF',
              font: {
                size: 14,
                family: 'Arial',
                weight: 'bold'
              }
            }
          }
        },
        plugins: {
          legend: {
            position: 'left',
            labels: {
              color: 'white',
              font: {
                size: 14,
                family: 'Arial',
                weight: 'bold'
              }
            }
          },
        }
      };
      
    return (
      <VStack spacing={8}>
        <VStack spacing={8}>
            <Heading as="h1" fontSize="6xl" fontWeight="bold" color="white" mt={6} mb={2}>
              Session History
            </Heading>
            <Text maxW="54rem">
                Click on the bar charts below to see a detailed session summary for each work session. 
            </Text>
            {chartData.labels && (
                        <Line data={chartData} options={options} />
            )}
        </VStack>
        <HStack spacing={10}>
            <Box width="100%" maxH="250px" overflowY="auto">
                <Table variant="simple" colorScheme="teal">
                    <Thead>
                    <Tr>
                        <Th color="white" fontFamily="Arial" fontWeight="bold">Date</Th>
                        <Th color="white" fontFamily="Arial" fontWeight="bold">View Details</Th>
                    </Tr>
                    </Thead>
                    <Tbody>
                    {sessions.map((session) => (
                        <Tr key={session.sessionId}>
                        <Td color="white">{session.timestamp}</Td>
                        <Td>
                            <Link color="teal.200" onClick={() => navigate(`/session-summary/${session.sessionId}`)}>
                            View Summary
                            </Link>
                        </Td>
                        </Tr>
                    ))}
                    </Tbody>
                </Table>
            </Box>
          </HStack>
      </VStack>
    ) 
     
    //   return (
    //     <VStack spacing={8}>
    //       <Heading as="h1" fontSize="6xl" fontWeight="bold" color="white" mt={6} mb={2}>
    //         Session History
    //       </Heading>
    //       <Text maxW="54rem">
    //         Click on the bar charts below to see a detailed session summary for each work session. 
    //       </Text>
    //       <HStack spacing={10}>
    //         {chartData.labels && (
    //             <Line data={chartData} options={options} />
    //         )}
    //         <Box width="100%" maxH="300px" overflowY="auto">
    //           <Table variant="simple" colorScheme="teal">
    //             <Thead>
    //               <Tr>
    //                 <Th color="white" fontFamily="Arial" fontWeight="bold">Date</Th>
    //                 <Th color="white" fontFamily="Arial" fontWeight="bold">View Details</Th>
    //               </Tr>
    //             </Thead>
    //             <Tbody>
    //               {sessions.map((session) => (
    //                 <Tr key={session.sessionId}>
    //                   <Td color="white">{session.timestamp}</Td>
    //                   <Td>
    //                     <Link color="teal.200" onClick={() => navigate(`/session-summary/${session.sessionId}`)}>
    //                       View Summary
    //                     </Link>
    //                   </Td>
    //                 </Tr>
    //               ))}
    //             </Tbody>
    //           </Table>
    //         </Box>
    //       </HStack>
    //     </VStack>
    //   ) 
}

export default SessionHistory;