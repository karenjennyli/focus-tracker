import React from 'react';
import { Heading } from '@chakra-ui/react';
import EventCard from './EventCard';
import './DetectionData.css';

const EventList = ({ events }) => {
  return (
    <div className="event-list-container">
      <Heading as="h1" fontSize="2xl" fontWeight="bold" color="white" mt={0} mb={4}>
        Latest Events
      </Heading>
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
