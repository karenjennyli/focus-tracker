import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';

import {
  VStack,
  Text,
  Heading,
} from '@chakra-ui/react';

function SessionHistory() {
    const [sessions, setSessions] = useState([]);

    useEffect(() => {
        fetch('http://127.0.0.1:8000/api/sessions')
            .then(response => response.json())
            .then(data => {
                // Handle data
                setSessions(data);
            })
            .catch(error => console.error('Error fetching sessions:', error));
    }, []);



    return (
      <VStack spacing={8}>
        <VStack spacing={8}>
            <Heading as="h1" fontSize="6xl" fontWeight="bold" color="white" mt={6} mb={2}>
              Session History
            </Heading>
            <Text maxW="54rem">
                Click on the graphs below to see a detailed session summary for each work session. 
            </Text>
            <div>
                {sessions.map(session => (
                    <div key={session.session_id}>
                        <Link to={`/session-summary/${session.session_id}`}>
                            <p>Session ID: {session.session_id}</p>
                        </Link>
                    </div>
                ))}
            </div>
        </VStack>
      </VStack>
    )
    
}

export default SessionHistory;