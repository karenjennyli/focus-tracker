import Plotly from 'plotly.js-basic-dist';
import createPlotlyComponent from 'react-plotly.js/factory';

const Plot = createPlotlyComponent(Plotly);

const DetectionsScatterPlot = ({ DetectionData, ProcessedFlowData, startTime }) => {
  // const min = startTime;
  // const max = new Date();
  const dates = DetectionData.map(d => new Date(d.timestamp));
  // const min = dates.length ? new Date(Math.min(...dates)) : startTime;
  // const max = dates.length ? new Date(Math.max(...dates)) : new Date();

  // Assuming dates is an array of date strings and startTime is a valid Date object
  const minDate = dates.length ? new Date(Math.min(...dates.map(date => new Date(date).getTime()))) : startTime;
  const maxDate = dates.length ? new Date(Math.max(...dates.map(date => new Date(date).getTime()))) : new Date();

  // Calculate the time difference
  const timeDiff = maxDate.getTime() - minDate.getTime();

  // Calculate 5% buffer on each side
  const buffer = timeDiff * 0.05;

  // Add and subtract the buffer from the max and min dates respectively
  const min = new Date(minDate.getTime() - buffer);
  const max = new Date(maxDate.getTime() + buffer);

  const chartData = [
    {
      x: [min],
      y: [10],  // Value outside the range of the y-axis
      mode: 'markers',
      name: 'Flow',
      marker: { color: 'green', size: 20, opacity: 0.25, symbol: 'square' },
      showlegend: true
    },
    {
      x: DetectionData.filter(d => d.detection_type === 'yawn').map(d => new Date(d.timestamp)),
      y: DetectionData.filter(d => d.detection_type === 'yawn').map(d => 1),
      mode: 'markers',
      name: 'Yawn',
      marker: { color: 'rgba(255, 99, 132, 0.5)', size: 10 },
      showlegend: false
    },
    {
      x: DetectionData.filter(d => d.detection_type === 'sleep').map(d => new Date(d.timestamp)),
      y: DetectionData.filter(d => d.detection_type === 'sleep').map(d => 2),
      mode: 'markers',
      name: 'Sleep',
      marker: { color: 'rgba(54, 162, 235, 0.5)', size: 10 },
      showlegend: false
    },
    {
      x: DetectionData.filter(d => d.detection_type.includes('gaze')).map(d => new Date(d.timestamp)),
      y: DetectionData.filter(d => d.detection_type.includes('gaze')).map(d => 3),
      mode: 'markers',
      name: 'Gaze',
      marker: { color: 'rgba(75, 192, 192, 0.5)', size: 10 },
      showlegend: false
    },
    {
      x: DetectionData.filter(d => d.detection_type.includes('phone')).map(d => new Date(d.timestamp)),
      y: DetectionData.filter(d => d.detection_type.includes('phone')).map(d => 4),
      mode: 'markers',
      name: 'Phone',
      marker: { color: 'rgba(255, 206, 86, 0.5)', size: 10 },
      showlegend: false
    },
    {
      x: DetectionData.filter(d => d.detection_type.includes('people')).map(d => new Date(d.timestamp)),
      y: DetectionData.filter(d => d.detection_type.includes('people')).map(d => 5),
      mode: 'markers',
      name: 'People',
      marker: { color: 'rgba(153, 102, 255, 0.5)', size: 10 },
      showlegend: false
    }
  ];

  const layout = {
    shapes: [
      ...ProcessedFlowData.filter(d => d.flow === 'Flow').map(d => ({
        type: 'rect',
        x0: new Date(d.timestamp_epoch * 1000),
        y0: 0,
        x1: new Date(d.timestamp_epoch * 1000).setSeconds(new Date(d.timestamp_epoch * 1000).getSeconds() + 5),
        y1: 6,
        fillcolor: 'green',
        opacity: 0.25,
        line: {
          color: 'green',
          width: 0
        }
      }))
    ],
    xaxis: {
      range: [min, max],
      type: 'time'
    },
    yaxis: {
      range: [0, 6],
      tickvals: [1, 2, 3, 4, 5],
      ticktext: ['Yawn', 'Sleep', 'Gaze', 'Phone', 'People']
    },
    font: {
      family: 'system-ui, sans-serif',
      size: 14
    },
    margin: {
      l: 75,  // left margin
      r: 50,  // right margin
      b: 75,  // bottom margin
      t: 50,  // top margin
      pad: 0  // padding
    },
    dragmode: false,
    autosize: false,
    height: 500
  };

  return (
    <Plot data={chartData} layout={layout} config={{displayModeBar: false}} style={{ width: '100%', height: '100%' }}/>
  );
};

export default DetectionsScatterPlot;
