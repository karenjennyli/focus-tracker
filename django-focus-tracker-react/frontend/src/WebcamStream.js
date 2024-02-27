import React, { useEffect, useRef } from 'react';

const WebcamStream = () => {
  const videoRef = useRef(null);

  useEffect(() => {
    const videoElement = videoRef.current; // Capture the current value of videoRef.current

    async function getDevices() {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        return devices.filter(device => device.kind === 'videoinput');
      } catch (error) {
        console.error('Error listing devices:', error);
        return [];
      }
    }

    async function selectExternalCamera(devices) {
      let externalCamera = devices.find(device => device.label.toLowerCase().includes('external'));
      // If no external camera is found, fallback to the default camera
      console.log("Default camera being used")
      return externalCamera ? { exact: externalCamera.deviceId } : true;
    }

    async function setupWebcam() {
      const devices = await getDevices();
      const videoSource = await selectExternalCamera(devices);

      const constraints = {
        video: { deviceId: videoSource },
      };

      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        if (videoElement) { // Use the captured value
          videoElement.srcObject = stream;
        }
      } catch (error) {
        console.error('Error accessing webcam:', error);
      }
    }

    setupWebcam();

    return () => {
      if (videoElement && videoElement.srcObject) { // Use the captured value in the cleanup function
        videoElement.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return <video ref={videoRef} autoPlay playsInline />;
};

export default WebcamStream;
