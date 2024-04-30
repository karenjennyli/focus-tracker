// import './App.css';
import React from 'react';
import { BrowserRouter as Router, Route, Routes} from 'react-router-dom';
import DetectionData from './DetectionInfo';
import CalibrationPage from './CalibrationPage';
import SessionSummary from './SessionSummary';
import FeatureComponent from './featureComponent';
import EmotiveHeadsetComponent from './emotiveHeadsetComponent';
import CameraCompontent from './cameraComponent';
import SessionHistory from './SessionHistory';
import { Link } from 'react-router-dom';
import { useEffect } from 'react';
import axios from 'axios';

import {
  ChakraProvider,
  Box,
  VStack,
  Grid,
  extendTheme,
  Text,
  Button,
  Flex,
  Container,
  Heading,
  useColorModeValue,
} from '@chakra-ui/react';

// Extend the theme to include custom extra colors and fonts
const colors = {
  brand: {
    blue: '#5D5DFF',
    black: '#1a202c',
  },
};

const theme = extendTheme({ colors });

function App() {
   const bgColor = useColorModeValue('brand.black', 'gray.800');
   const color = useColorModeValue('white', 'gray.200');

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
 
  return (
    <ChakraProvider theme={theme}>
      <Router>
      <Box textAlign="center" fontSize="xl" bg={bgColor} color={color} minHeight="100vh">
        <Grid p={3} pt={0}>
          {/* Navigation Bar */}
          <Container maxW="1400px" mx="auto" px={0}>
          <Flex
            as="header"
            width="full"
            align="center"
            justifyContent="space-between"
            pt={5} // reduce top padding
            pb={2} // reduce bottom padding
          >
            <Text fontSize="xlg" fontWeight="bold" color="white">
              MindFlow
            </Text>
            <Flex>
              <Link to="/session-history">
                <Button colorScheme="blue" width="90px">History</Button>
              </Link>
              <div style={{ width: '10px' }}></div>
              <Link to="/">
                <Button colorScheme="blue" width="90px">Home</Button>
              </Link>
            </Flex>
          </Flex>
          </Container>

          <Routes> {/* Use Routes instead of Switch */}
            <Route path="/" element={<MainContent />} />
            <Route path="/calibration-page" element={<CalibrationPage />} />
            <Route path="/detection-info" element={<DetectionData />} />
            <Route path="/session-summary/:sessionIDFromURL" element={<SessionSummary />} />
            <Route path="/session-history" element={<SessionHistory />} />
          </Routes>
        </Grid>
      </Box>
      </Router>
    </ChakraProvider>
  );
}

function MainContent() {
  return (
    <VStack spacing={8}>
      <VStack spacing={8}>
          <Heading as="h1" fontSize="8xl" fontWeight="bold" color="brand.blue" mt={6} mb={2}>
            Focus Tracker App
          </Heading>
          <Text maxW="54rem">
            This app leverages an EEG headset and web camera, employing machine learning algorithms to accurately measure users' focus levels and identify distractions in real time during work sessions. It aims to help users identify actionable steps to enhance their productivity.
          </Text>
      </VStack>

      {/* Compontents Section */}    
      <FeatureComponent/>
      <EmotiveHeadsetComponent/>
      <CameraCompontent/>

      {/* Footer */}
      <Flex as="footer" width="full" align="center" justifyContent="center" p={10}>
        <Text>© 2024 Focus Tracker App</Text>
      </Flex>
    </VStack>
  );
}

export default App;
