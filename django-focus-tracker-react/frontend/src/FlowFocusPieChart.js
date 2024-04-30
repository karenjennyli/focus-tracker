import { Pie } from 'react-chartjs-2';

const FlowFocusPieChart = ({ FlowData, FocusData, flowFocus }) => {

  const options = {
    plugins: {
      legend: {
        labels: {
          color: 'white',
          font: {
            size: 13,
            weight: 'bold',
            family: 'system-ui, sans-serif'
          }
        }
      }
    }
  };

  return (
    <Pie
      data={{
        labels: flowFocus === 'Flow' ? ['Flow', 'Not in Flow'] : ['Focus', 'Not in Focus'],
        datasets: [{
          data: flowFocus === 'Flow' ? [FlowData[0].flowCount, FlowData[0].notInFlowCount] : [FocusData[0].focusCount, FocusData[0].notInFocusCount],
          backgroundColor: [
            flowFocus === 'Flow' ? '#b9d8a8' : '#cba4df',
            '#d9d7d7'
          ],
        }]
      }}
      options={options}
    />
  );
}
 
export default FlowFocusPieChart;
