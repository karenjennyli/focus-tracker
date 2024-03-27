import React from 'react';
import { Link } from 'react-router-dom';
import './CalibrationPage.css';

function CalibrationPage() {
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