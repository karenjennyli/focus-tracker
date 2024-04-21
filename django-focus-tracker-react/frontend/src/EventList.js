import React from 'react';
import { Heading } from '@chakra-ui/react';
import EventCard from './EventCard';
import './DetectionData.css';

// Helper function to format timestamp
const parseAndFormatTime = (timestamp) => {
  // Extract the time part (HH:MM:SS) from the timestamp
  const timePart = timestamp.split('T')[1].split('Z')[0];
  let [hours, minutes] = timePart.split(':');

  // Convert hours to number 
  hours = parseInt(hours, 10);

  // Determine AM or PM
  const ampm = hours >= 12 ? 'PM' : 'AM';

  // Convert to 12-hour format
  hours = hours % 12;
  hours = hours ? hours : 12; // the hour '0' should be '12'

  // Return formatted time string
  return `${hours}:${minutes} ${ampm}`;
};

const EventList = ({ detectionData, displayTitle }) => {
  const eventsList = detectionData.map((event) => ({
    timestamp: parseAndFormatTime(event.timestamp),
    eventType: event.detection_type,
    imageUrl: event.image_url,
    info: event.aspect_ratio,
  }));

  return (
    <div className={displayTitle ? 'event-list-container' : 'summary-event-list-container'}>
      {displayTitle && (
            <Heading as="h1" fontSize="2xl" fontWeight="bold" color="white" mt={0} mb={4}>
            Latest Events
          </Heading>  
      )}
      <div className="event-list">
        {eventsList.length === 0 && <p>No detected events.</p>}
        {eventsList.map((event, index) => (
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
