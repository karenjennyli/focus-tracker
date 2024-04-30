import React, { useEffect, useState } from 'react';
import { Bar } from 'react-chartjs-2';
import { useNavigate } from 'react-router-dom';

import {
  VStack,
  Text,
  Heading,
} from '@chakra-ui/react';

function SessionHistory() {
    const [chartData, setChartData] = useState({});
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
                setChartData({
                    labels: sessionData.map(item => item.timestamp),
                    datasets: [
                        {
                            label: 'Total Distractions',
                            data: sessionData.map(item => item.distractions),
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                            borderColor: 'rgba(75, 192, 192, 1)',
                            borderWidth: 3,
                        }
                    ],
                    sessionIds: sessionData.map(item => item.sessionId)
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
                <Bar data={chartData} options={options} />
            )}
        </VStack>
      </VStack>
    )
    
}

export default SessionHistory;