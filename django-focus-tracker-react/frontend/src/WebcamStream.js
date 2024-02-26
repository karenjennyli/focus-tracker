import React, { useEffect, useRef } from 'react';

const WebcamStream = () => {
  const videoRef = useRef(null);

  useEffect(() => {
    const getVideo = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        // Handle the error
        console.error("Error accessing the webcam", err);
      }
    };

    getVideo();
  }, [videoRef]);

  return <video ref={videoRef} autoPlay playsInline />;
};

export default WebcamStream;
