import React from "react";
import { PieChart, Pie, Tooltip } from "recharts";

const data01 = [
  { name: "Focus Score A", value: 70 },
  { name: "Focus Score B", value: 100 },
  { name: "Focus Score C", value: 84 },
  { name: "Focus Score D", value: 58 },
  { name: "Focus Score E", value: 43 },
  { name: "Focus Score F", value: 72 }
];


export default function PChart() {
  return (
    <PieChart width={1000} height={400}>
      <Pie
        dataKey="value"
        isAnimationActive={false}
        data={data01}
        cx={200}
        cy={200}
        outerRadius={80}
        fill="#990011"
        label
      />
      <Tooltip />
    </PieChart>
  );
}
