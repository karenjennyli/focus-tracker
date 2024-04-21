import { Pie } from 'react-chartjs-2';

const FlowPieChart = ({ FlowData }) => {

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
        labels: ['Flow', 'Not in Flow'],
        datasets: [{
          data: [FlowData[0].flowCount, FlowData[0].notInFlowCount],
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
 
export default FlowPieChart;