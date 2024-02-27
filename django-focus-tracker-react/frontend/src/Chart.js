import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend
} from "recharts";

const data = [
  {
    name: "12:00",
    Focus_Level: 100,
    Distractions: 1,
  },
  {
    name: "12:10",
    Focus_Level: 85,
    Distractions: 3,
  },
  {
    name: "12:30",
    Focus_Level: 72,
    Distractions: 4,
  },
  {
    name: "12:45",
    Focus_Level: 83,
    Distractions: 2,
  },
  {
    name: "1:00",
    Focus_Level: 86,
    Distractions: 2,
  },
];

export default function Chart() {
  return (
    <BarChart
      width={500}
      height={300}
      data={data}
      margin={{
        top: 20,
        right: 30,
        left: 20,
        bottom: 5
      }}
    >
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="name" stroke="#000000" />
      <YAxis yAxisId="left" orientation="left" stroke="#000000" />
      <YAxis yAxisId="right" orientation="right" stroke="#000000" />
      <Tooltip />
      <Legend />
      <Bar yAxisId="left" dataKey="Focus_Level" fill="#8884d8" />
      <Bar yAxisId="right" dataKey="Distractions" fill="#027148" />
    </BarChart>
  );
}
