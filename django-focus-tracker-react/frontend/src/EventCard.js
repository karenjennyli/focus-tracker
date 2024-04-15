import React from 'react';
import './EventList.css';

const baseURL = 'http://127.0.0.1:8000';

const eventTypeDescriptions = {
  'gaze left': 'User looking left away from the screen',
  'gaze right': 'User looking right away from the screen',
  'yawn': 'User yawning',
  'sleep': 'User\'s eyes closed for an extended period, indicating microsleep',
  'people': 'Other people detected in frame as a distraction',
  'phone': 'User interacting with phone',
  'user not recognized': 'User is no longer recognized in frame',
  'user returned': 'User has returned to frame'
};

const getDescription = (eventType, info) => {
  if (eventType === 'user returned') {
    // if info > 60 seconds, return as min and sec
    // else return as seconds
    const minutes = Math.floor(info / 60);
    const seconds = Math.floor(info % 60);
    if (info > 60) {
      return 'User has returned to frame after ' + minutes + ' minutes and ' + seconds + ' seconds';
    } else {
      return 'User has returned to frame after ' + seconds + ' seconds';
    }
  }
  return eventTypeDescriptions[eventType] || 'Unknown event';
}

const EventCard = ({ timestamp, eventType, imageUrl, info }) => {
  const description = getDescription(eventType, info);

  return (
    <div className="event-card">
      <img src={baseURL + imageUrl} alt="Event Thumbnail" className="thumbnail" />
      <div className="event-footer">
        <span className="timestamp">{timestamp}</span>
        <div className="description">{description}</div>
      </div>
    </div>
  );
};

export default EventCard;
