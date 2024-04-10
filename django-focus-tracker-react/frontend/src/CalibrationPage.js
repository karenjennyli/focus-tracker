import React, { useEffect } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import './CalibrationPage.css';

import {
  VStack,
  Text,
  Heading,
} from '@chakra-ui/react';

function CalibrationPage() {

    useEffect(() => {
        // Function to call the Django backend
        async function startCalibration() {
          try {
            await axios.post('http://127.0.0.1:8000/api/start_calibration/');
            console.log('Calibration started');
          } catch (error) {
            console.error('Error starting calibration:', error);
          }
        }
    
        startCalibration();
      }, []);

    return (
      <VStack spacing={8}>
        <VStack spacing={8}>
            <Heading as="h1" fontSize="6xl" fontWeight="bold" color="white" mt={6} mb={2}>
              Calibration
            </Heading>
            <Text maxW="54rem">
              Look into the web camera and adjust the Emotiv Headset until calibration is complete. <br></br> Press the green button below to continue.
            </Text>
            <div className="container-fixed-bottom">
                <Link to="/detection-info">
                    <button className="checkmark-button"></button>
                </Link>
            </div>  
        </VStack>
      </VStack>
    )
    
}

export default CalibrationPage;