import React, { useEffect } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import './CalibrationPage.css';

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
        <div className="calibration-content">
            <h1>Calibration</h1>
            <div className="calibration-intro">
                <p>Look into the web camera and adjust the Emotiv Headset until calibration is complete. <br></br>Press the green button below to continue.</p>
                <br></br>
                <div className="container-fixed-bottom">
                    <Link to="/detection-info">
                        <button className="checkmark-button"></button>
                    </Link>
                </div>
            </div>       
        </div>
    )
}

export default CalibrationPage;