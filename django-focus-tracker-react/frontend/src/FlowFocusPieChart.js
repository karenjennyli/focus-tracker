import { Pie } from 'react-chartjs-2';

const FlowFocusPieChart = ({ FlowData, FocusData, flowFocus }) => {

  const options = {
    plugins: {
      legend: {
        labels: {
          color: 'white',
          font: {
            size: 14,
            weight: 'bold'
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
            'green',
            'rgba(192, 75, 75, 1)',
          ],
        }]
      }}
      options={options}
    />
  );
}
 
export default FlowFocusPieChart;