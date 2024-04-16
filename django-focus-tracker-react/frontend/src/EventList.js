import React from 'react';
import EventCard from './EventCard';
import './DetectionData.css';

const EventList = ({ events }) => {
  return (
    <div className="event-list-container">
      <div className="event-list">
        {events.length === 0 && <p>No detected events.</p>}
        {events.map((event, index) => (
          <EventCard
            key={index}
            timestamp={event.timestamp}
            eventType={event.eventType}
            imageUrl={event.imageUrl}
            info={event.info}
          />
        ))}
      </div>
    </div>
  );
};

export default EventList;
