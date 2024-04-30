import { Bar } from 'react-chartjs-2';

const DetectionsBarChart = ({ lastDetectionData }) => {
    const options = {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true,
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1
                },
                ticks: {
                    color: '#FFF',
                    font: {
                        size: 14,
                        weight: 'bold'
                    },
                    stepSize: 1
                }
            },
            x: {
                ticks: {
                    color: '#FFF',
                    font: {
                        size: 14,
                        weight: 'bold'
                    }
                },
            },
        },
        plugins: {
            legend: {
              labels: {
                color: 'white',
                font: {
                  size: 14,
                  weight: 'bold',
                  family: 'system-ui, sans-serif'
                }
              }
            },
        }
    };

  return (
    <Bar data={
        {
            labels: lastDetectionData.map(item => item.detection_type.charAt(0).toUpperCase() + item.detection_type.slice(1)),
        datasets: [
                {
                    label: 'Frequency',
                    data: lastDetectionData.map(item => item.frequency),
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 3
                }
            ]
        }
    }
    options={options}
    />
  );
}
 
export default DetectionsBarChart;