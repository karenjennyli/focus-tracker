// import './App.css';
import React from 'react';
import { BrowserRouter as Router, Route, Routes} from 'react-router-dom';
import DetectionData from './DetectionInfo';
import CalibrationPage from './CalibrationPage';
import SessionSummary from './SessionSummary';
import FeatureComponent from './featureComponent';
import EmotiveHeadsetComponent from './emotiveHeadsetComponent';
import CameraCompontent from './cameraComponent';

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
 
  return (
    <ChakraProvider theme={theme}>
      <Router>
      <Box textAlign="center" fontSize="xl" bg={bgColor} color={color}>
        <Grid minH="100vh" p={3} pt={0}>
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
              <Text fontSize="lg" fontWeight="bold" color="white">
                Logo
              </Text>
              <Button colorScheme="blue">Home</Button>
            </Flex>
          </Container>

          <Routes> {/* Use Routes instead of Switch */}
          <Route path="/" element={<MainContent />} />
            <Route path="/calibration-page" element={<CalibrationPage />} />
            <Route path="/detection-info" element={<DetectionData />} />
            <Route path="/session-summary" element={<SessionSummary />} />
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
