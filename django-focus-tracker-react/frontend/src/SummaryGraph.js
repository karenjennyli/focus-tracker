import React from 'react';
import { Scatter } from 'react-chartjs-2';

const SummaryGraph = ( { DetectionData, startTime } ) => {

  const min = startTime;
  const max = new Date();

    // Prepare chart data
    const chartData = {
      datasets: [
          {
              label: 'Yawn',
              data: DetectionData.filter(d => d.detection_type === 'yawn')
                  .map(d => ({ x: new Date(d.timestamp), y: 1 })),
              backgroundColor: 'rgba(255, 99, 132, 0.5)',
              borderColor: 'rgba(0, 0, 0, 0.8)', // Make lines darker,
              pointRadius: 5,
          },
          {
              label: 'Sleep',
              data: DetectionData.filter(d => d.detection_type === 'sleep')
                  .map(d => ({ x: new Date(d.timestamp), y: 2 })),
              backgroundColor: 'rgba(54, 162, 235, 0.5)',
              borderColor: 'rgba(0, 0, 0, 0.8)', // Make lines darker,
              pointRadius: 5
          },
          {
              label: 'Gaze',
              data: DetectionData.filter(d => d.detection_type.includes('gaze'))
                  .map(d => ({ x: new Date(d.timestamp), y: 3 })),
              backgroundColor: 'rgba(75, 192, 192, 0.5)',
              borderColor: 'rgba(0, 0, 0, 0.8)', // Make lines darker,
              pointRadius: 5
          },
          {
              label: 'Phone',
              data: DetectionData.filter(d => d.detection_type.includes('phone'))
                  .map(d => ({ x: new Date(d.timestamp), y: 4 })),
              backgroundColor: 'rgba(255, 206, 86, 0.5)',
              borderColor: 'rgba(0, 0, 0, 0.8)', // Make lines darker,
              pointRadius: 5
          },
          {
              label: 'People',
              data: DetectionData.filter(d => d.detection_type.includes('people'))
                  .map(d => ({ x: new Date(d.timestamp), y: 5 })),
              backgroundColor: 'rgba(153, 102, 255, 0.5)',
              borderColor: 'rgba(0, 0, 0, 0.8)', // Make lines darker,
              pointRadius: 5
          }
      ]
  };

  const options = {
    scales: {
        x: {
            grid: {
                display: false
            },
            type: 'time',
            time: {
                unit: 'minute',
                tooltipFormat: 'h:mm',
                displayFormats: {
                    minute: 'h:mm' // Apply the same format for axis labels
                },
                round: 'second'
            },
            title: {
                display: false,
                text: 'Time',
                color: '#000', // Dark font color
                font: {
                    size: 16 // Larger font size
                }
            },
            ticks: {
                color: '#000', // Dark font color for ticks
                font: {
                    size: 14 // Larger font size for ticks
                },
                stepSize: 1
            },
            min: min,
            max: max
        },
        y: {
            grid: {
                display: false
            },
            ticks: {
                // This will only work if you have numeric values for y
                callback: function (value) {
                    if (value === 1) return 'Yawn';
                    else if (value === 2) return 'Sleep';
                    else if (value === 3) return 'Gaze';
                    else if (value === 4) return 'Phone';
                    else if (value === 5) return 'People';
                    return null;
                },
                color: '#000', // Dark font color for ticks
                font: {
                    size: 14 // Larger font size for ticks
                }
            },
            title: {
                display: false,
                text: 'Distraction Type', 
                color: '#000', // Dark font color
                font: {
                    size: 16 // Larger font size
                }
            },
            min: 0,
            max: 6
        }
    },
    plugins: {
        legend: {
            labels: {
                color: '#000', // Dark font color for legend
                font: {
                    size: 14 // Larger font size for legend
                }
            },
            position: 'top',
        }
    }
};

  return (
    DetectionData.length > 0 && (
      <Scatter data={chartData} options={options}/>
  ));
};

export default SummaryGraph;
