import React from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend
} from "recharts";

const data01 = [
  { x: 100, y: 200},
  { x: 120, y: 100},
  { x: 170, y: 300},
  { x: 140, y: 250},
  { x: 150, y: 400},
  { x: 110, y: 280}
];
const data02 = [
  { x: 200, y: 260},
  { x: 240, y: 290},
  { x: 190, y: 290},
  { x: 198, y: 250},
  { x: 180, y: 280},
  { x: 210, y: 220}
];

export default function LChart() {
  return (
    <ScatterChart
      width={400}
      height={300}
      margin={{
        top: 20,
        right: 30,
        bottom: 5,
        left: 20
      }}
    >
      <CartesianGrid />
      <XAxis type="number" dataKey="x" name="time" unit="" stroke="#000000" />
      <YAxis type="number" dataKey="y" name="distraction" unit="" stroke="#000000" />
      <Tooltip cursor={{ strokeDasharray: "3 3" }} />
      <Legend />
      <Scatter name="Focused" data={data01} fill="#000000" shape="square" />
      <Scatter name="Not Focused" data={data02} fill="#FFFF00" shape="triangle" />
    </ScatterChart>
  );
}
