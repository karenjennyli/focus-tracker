import React from 'react';
import './EventCard.css';

const baseURL = 'http://127.0.0.1:8000';

const eventTypeDescriptions = {
  'gaze left': 'User looking left away from the screen',
  'gaze right': 'User looking right away from the screen',
  'yawn': 'User yawning',
  'sleep': 'User\'s eyes closed for an extended period, indicating microsleep',
  'people': 'Other people detected in frame as a distraction',
  'phone': 'User interacting with phone',
  'face not recognized': 'User\'s face is no longer recognized'
};

const EventCard = ({ timestamp, eventType, imageUrl }) => {
  const description = eventTypeDescriptions[eventType] || 'Unknown event';

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
