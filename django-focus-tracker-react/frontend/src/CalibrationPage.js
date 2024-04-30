import React, { useEffect } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import './CalibrationPage.css';
import Webcam from "react-webcam";

import {
  VStack,
  Text,
  Heading,
} from '@chakra-ui/react';

function CalibrationPage() {

  const videoConstraints = {
    width: 1920,
    height: 1080,
    facingMode: "user",
    deviceId: "e4fc9040a6bbd234b0da54ee7c9e5e1796b5ced07398f1ae418c9139b56beb69"
  }

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
      <VStack spacing={8}>
        <VStack spacing={8}>
            <Heading as="h1" fontSize="6xl" fontWeight="bold" color="white" mt={6} mb={2}>
              Calibration
            </Heading>
            <Text maxW="54rem">
              Look into the web camera and adjust the Emotiv Headset until calibration is complete. <br></br> Press the green button below to continue.
            </Text>
            <Webcam className='webcam-calibration' audio={false} mirrored={true} videoConstraints={videoConstraints}/>
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