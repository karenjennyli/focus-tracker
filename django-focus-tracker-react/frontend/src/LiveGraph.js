import React, { useEffect, useState } from 'react';
import Plotly from 'plotly.js-basic-dist';
import createPlotlyComponent from 'react-plotly.js/factory';

const Plot = createPlotlyComponent(Plotly);

const LiveGraph = ({ DetectionData }) => {

  const [min, setMin] = useState(new Date(Date.now() - 10 * 60 * 1000));
  const [max, setMax] = useState(new Date());

  useEffect(() => {
    const intervalId = setInterval(() => {
      setMin(new Date(Date.now() - 10 * 60 * 1000));
      setMax(new Date());
    }, 1000);

    return () => clearInterval(intervalId);
  }, []);

  const chartData = [
    {
      x: DetectionData.filter(d => d.detection_type === 'yawn').map(d => new Date(d.timestamp)),
      y: DetectionData.filter(d => d.detection_type === 'yawn').map(d => 1),
      mode: 'markers',
      name: 'Yawn',
      marker: { color: 'rgba(255, 99, 132, 0.5)', size: 10 }
    },
    {
      x: DetectionData.filter(d => d.detection_type === 'sleep').map(d => new Date(d.timestamp)),
      y: DetectionData.filter(d => d.detection_type === 'sleep').map(d => 2),
      mode: 'markers',
      name: 'Sleep',
      marker: { color: 'rgba(54, 162, 235, 0.5)', size: 10 }
    },
    {
      x: DetectionData.filter(d => d.detection_type.includes('gaze')).map(d => new Date(d.timestamp)),
      y: DetectionData.filter(d => d.detection_type.includes('gaze')).map(d => 3),
      mode: 'markers',
      name: 'Gaze',
      marker: { color: 'rgba(75, 192, 192, 0.5)', size: 10 }
    },
    {
      x: DetectionData.filter(d => d.detection_type.includes('phone')).map(d => new Date(d.timestamp)),
      y: DetectionData.filter(d => d.detection_type.includes('phone')).map(d => 4),
      mode: 'markers',
      name: 'Phone',
      marker: { color: 'rgba(255, 206, 86, 0.5)', size: 10 }
    },
    {
      x: DetectionData.filter(d => d.detection_type.includes('people')).map(d => new Date(d.timestamp)),
      y: DetectionData.filter(d => d.detection_type.includes('people')).map(d => 5),
      mode: 'markers',
      name: 'People',
      marker: { color: 'rgba(153, 102, 255, 0.5)', size: 10 }
    }
  ];

  const layout = {
    xaxis: {
      range: [min, max],
      type: 'date'
    },
    yaxis: {
      range: [0, 6],
      tickvals: [1, 2, 3, 4, 5],
      ticktext: ['Yawn', 'Sleep', 'Gaze', 'Phone', 'People']
    },
    title: 'Live Graph'
  };

  return (
    <Plot data={chartData} layout={layout} />
  );
};

export default LiveGraph;
