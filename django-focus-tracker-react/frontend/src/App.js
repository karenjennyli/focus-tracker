// import './App.css';
import React from 'react';
import { BrowserRouter as Router, Route, Routes, useNavigate} from 'react-router-dom';
import DetectionData from './DetectionInfo';
import CalibrationPage from './CalibrationPage';
import SessionSummary from './SessionSummary';
import emotiveHeadset from './emotiveHeadset.png'

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
              pt={3} // reduce top padding
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
          <Heading as="h1" size="4xl" fontWeight="bold" color="brand.blue" mt={0} mb={2}>
            Focus Tracker App
          </Heading>
          <Text maxW="46rem">
            This App enables users to measure their focus and associated distractions during work sessions to help them identify actionable steps to improve their productivity.
          </Text>
          <Button size="lg" colorScheme="blue" onClick={() => navigate('/calibration-page')}>
            Get Started
          </Button>
      </VStack>
      {/* Features Section */}
      <Container maxW="container.xl" pt={10}>
        <Box p={5} shadow="md" borderWidth="1px" borderRadius="md" overflow="hidden">
          <Flex align="center" justify="center" direction={{ base: "column", md: "row" }}>
            <Box flexShrink={0}>
              <Image
                src={emotiveHeadset}
                alt="Emotiv Headset"
                boxSize="500px" 
                objectFit="contain"
                p={2} // Padding to ensure the image doesn't touch the edges of the box
              />
            </Box>
            <Box ml={{ md: 6 }} mt={{ base: 4, md: 0 }}>
              <Text fontWeight="bold" textTransform="uppercase" fontSize="lg" letterSpacing="wide" color="teal.600">
                Feature One
              </Text>
              <Text mt={2} color="gray.600">
                Detailed explanation of the feature that relates to the image of the Emotiv headset.
              </Text>
              {/* Add more descriptive text or buttons as needed */}
            </Box>
          </Flex>
        </Box>
      </Container>
      {
        /* 
         <Flex direction={{ base: "column", md: "row" }} justify="space-around">
          <Box p={5} shadow="md" borderWidth="1px">
            <Heading fontSize="xl">Feature One</Heading>
            <Text mt={4}>Description for feature one.</Text>
          </Box>
          <Box p={5} shadow="md" borderWidth="1px">
            <Heading fontSize="xl">Feature Two</Heading>
            <Text mt={4}>Description for feature two.</Text>
          </Box>
        </Flex>*/
      }

      {/* Footer */}
      <Flex as="footer" width="full" align="center" justifyContent="center" p={4}>
        <Text>© 2024 Your Company</Text>
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
