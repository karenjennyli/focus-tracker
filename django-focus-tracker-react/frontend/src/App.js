import './App.css';
import React from 'react';
import { BrowserRouter as Router, Route, Routes, useNavigate, Link} from 'react-router-dom';
import Chart from "./Chart";
import PChart from "./PieChart";
import LChart from "./LineChart";
// import WebcamStream from './WebcamStream';
import DetectionData from './DetectionInfo';
import CalibrationPage from './CalibrationPage';
import SessionSummary from './SessionSummary';
import dashboardImage from './dashboard.jpg';

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
  Spacer,
  Container,
  Heading,
  useColorModeValue,
} from '@chakra-ui/react';

// Extend the theme to include custom colors, fonts, etc.
const colors = {
  brand: {
    blue: '#5D5DFF', // Replace with the exact blue color from the screenshot
    black: '#1a202c', // for the black background
    // Add other colors from the screenshot as needed
  },
};

const theme = extendTheme({ colors });

function App() {
  // Use the useColorModeValue hook to switch colors based on the theme
   const bgColor = useColorModeValue('brand.black', 'gray.800');
   const color = useColorModeValue('white', 'gray.200');
 
  return (
    <ChakraProvider theme={theme}>
      <Router>
      <Box textAlign="center" fontSize="xl" bg={bgColor} color={color}>
        <Grid minH="100vh" p={3}>
          {/* Navigation Bar */}
          <Flex as="header" width="full" align="center" justifyContent="space-between" p={4}>
            <Text fontSize="lg" fontWeight="bold" color="white">
              YourLogo
            </Text>
            <Button colorScheme="blue">Get Started</Button>
          </Flex>

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
  const navigate = useNavigate(); // useNavigate is now called within a component that is a child of <Router>

  return (
    <VStack spacing={8}>
      <VStack spacing={8}>
          <Heading as="h1" size="4xl" fontWeight="bold" color="brand.blue">
            Focus Tracker App
          </Heading>
          <Text maxW="46rem">
            This App enables users to measure their focus and associated distractions during work sessions to help them identify actionable steps to improve their productivity.
          </Text>
          <Button size="lg" colorScheme="blue" onClick={() => navigate('/calibration-page')}>
            Get Started
          </Button>
          {/* Assuming you want to keep the same image. If not, set the correct path to the hero image. */}
          {/* <Image
            borderRadius="md"
            src={dashboardImage}
            alt="Dashboard"
          /> */}
      </VStack>
      {/* Features Section */}
      <Container maxW="container.xl" pt={10}>
        <Flex direction={{ base: "column", md: "row" }} justify="space-around">
          <Box p={5} shadow="md" borderWidth="1px">
            <Heading fontSize="xl">Feature One</Heading>
            <Text mt={4}>Description for feature one.</Text>
          </Box>
          <Box p={5} shadow="md" borderWidth="1px">
            <Heading fontSize="xl">Feature Two</Heading>
            <Text mt={4}>Description for feature two.</Text>
          </Box>
          {/* Add more features as needed */}
        </Flex>
      </Container>

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
