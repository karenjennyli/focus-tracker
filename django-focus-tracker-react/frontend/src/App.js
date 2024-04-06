// import './App.css';
import React from 'react';
import { BrowserRouter as Router, Route, Routes, useNavigate} from 'react-router-dom';
import DetectionData from './DetectionInfo';
import CalibrationPage from './CalibrationPage';
import SessionSummary from './SessionSummary';
import emotiveHeadset from './emotiveHeadset.png'
import camera from './camera2.png'

// function App() {
//   return (
//     <Router>
//       <div className="App">
//         <Header />
//         <Routes> {/* Use Routes instead of Switch */}
//           <Route path="/" element={<MainContent />} />
//           <Route path="/calibration-page" element={<CalibrationPage />} />
//           <Route path="/detection-info" element={<DetectionData />} />
//           <Route path="/session-summary" element={<SessionSummary />} />
//         </Routes>
//       </div>
//     </Router>
//   );
// }

import {
  ChakraProvider,
  Box,
  VStack,
  Grid,
  extendTheme,
  Text,
  Button,
  Flex,
  Image,
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
  const navigate = useNavigate();

  return (
    <VStack spacing={8}>
      <VStack spacing={8}>
          <Heading as="h1" size="4xl" fontWeight="bold" color="brand.blue" mt={6} mb={2}>
            Focus Tracker App
          </Heading>
          <Text maxW="54rem">
            This app leverages an EEG headset and web camera, employing machine learning algorithms to accurately measure users' focus levels and identify distractions in real time during work sessions. It aims to help users identify actionable steps to enhance their productivity.
          </Text>
          <Button mt={3} size="lg" colorScheme="blue" onClick={() => navigate('/calibration-page')}>
            Get Started
          </Button>
      </VStack>
      {/* Features Section */}
      <Flex direction={{ base: "column", lg: "row" }} justify="space-between" pt={10}>
        <Container maxW="container.xl">
          <Box p={5} shadow="md" borderWidth="1px" borderRadius="md" overflow="hidden" bg="gray.500">
            <Flex align="center" justify="center" direction={{ base: "column", md: "row" }}>
              <Box flexShrink={0}>
                <Image
                  src={emotiveHeadset}
                  alt="Emotiv Headset"
                  boxSize="350px" 
                  objectFit="contain"
                  p={2} // Padding to ensure the image doesn't touch the edges of the box
                />
              </Box>
              <Box ml={{ md: 6 }} mt={{ base: 4, md: 0 }}>
                <Text fontWeight="bold" textTransform="uppercase" fontSize="lg" letterSpacing="wide" color="teal.300">
                  Emotiv Insight EEG Headset
                </Text>
                <Text mt={2} color="gray.900">
                  The Insight headset has 5 sensors which places it right in the middle of other EEG headset options which have anywhere from 1-32 sensors to measure EEG signals from different regions of the brain.
                </Text>
                {/* Add more descriptive text here if needed */}
              </Box>
            </Flex>
          </Box>
        </Container>

        <Container maxW="container.xl">
          <Box p={5} shadow="md" borderWidth="1px" borderRadius="md" overflow="hidden" bg="gray.500">
            <Flex align="center" justify="center" direction={{ base: "column", md: "row" }}>
              <Box flexShrink={0}>
                <Image
                  src={camera}
                  alt="Camera"
                  boxSize="350px" 
                  objectFit="contain"
                  p={2} // Padding to ensure the image doesn't touch the edges of the box
                />
              </Box>
              <Box ml={{ md: 6 }} mt={{ base: 4, md: 0 }}>
                <Text fontWeight="bold" textTransform="uppercase" fontSize="lg" letterSpacing="wide" color="teal.300">
                  Camera
                </Text>
                <Text mt={2} color="gray.900">
                  The TedGem 1080p camera has a processing rate 10 fps and ensures high-quality real-time monitoring of physical indicators of loss of focus, such as yawning, microsleeps, off-screen gazing, interruptions, and phone pick-ups. 
                </Text>
                {/* Add more descriptive text here if needed */}
              </Box>
            </Flex>
          </Box>
        </Container>  
      </Flex>

      {/* Footer */}
      <Flex as="footer" width="full" align="center" justifyContent="center" p={10}>
        <Text>© 2024 Focus Tracker App</Text>
      </Flex>
    </VStack>
  );
}


// function Header() {
//   return (
//     <header className="App-header">
//       <div className="logo">Logo</div>
//       <nav>
//         <ul>
//           <li>Home</li>
//           <li>Features</li>
//           <li>About</li>
//         </ul>
//       </nav>
//     </header>
//   );
// }

// function MainContent() {
//   return (
//     <div className="main-content">
//       <h1>Focus Tracker App</h1>
//       <div className="intro">
//         <p>This App enables users to measure their focus and associated distractions during <br></br>work sessions to help them identify actionable steps to improve their productivity.</p>
//         <br></br>
//         <Link to="/calibration-page"><button className="get-started">Get Started</button></Link>
//       </div>
//       <div className="charts-container">
//         <div className="chart">
//           <Chart />
//         </div>
//         <div className="pchart">
//           <PChart />
//         </div>
//         <div className="lchart">
//           <LChart />
//         </div>
//       </div>
//     </div>
//   );
// }

export default App;
