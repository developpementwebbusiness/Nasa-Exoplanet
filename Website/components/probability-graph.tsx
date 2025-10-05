"use client"

import { motion } from "framer-motion"
import {
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from "recharts"
import { Card } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { TrendingUp, ScanTextIcon as ScatterIcon } from "lucide-react"

interface ProbabilityGraphProps {
  predictions: any[]
  onSelectCandidate: (index: number) => void
  selectedCandidate: number | null
}

export function ProbabilityGraph({ predictions, onSelectCandidate, selectedCandidate }: ProbabilityGraphProps) {
  if (predictions.length === 0) {
    return (
      <Card className="p-8 text-center bg-card border-border">
        <p className="text-muted-foreground">Upload CSV data to see probability analysis</p>
      </Card>
    )
  }

  const chartData = predictions.map((pred, index) => ({
    index: index + 1,
    probability: (pred.probability || 0) * 100,
    isSelected: index === selectedCandidate,
  }))

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      const prob = payload[0].value
      const classification = prob >= 90 ? "Confirmed" : prob >= 50 ? "Likely" : prob >= 10 ? "Unsure" : "Unlikely"

      return (
        <div className="bg-card border-2 border-primary rounded-lg p-3 shadow-xl">
          <p className="text-sm font-semibold text-foreground">Candidate #{data.index}</p>
          <p className="text-sm text-primary font-medium">Probability: {prob.toFixed(2)}%</p>
          <p className="text-xs text-muted-foreground mt-1">Classification: {classification}</p>
        </div>
      )
    }
    return null
  }

  const CustomDot = (props: any) => {
    const { cx, cy, payload } = props
    const isSelected = payload.index - 1 === selectedCandidate

    return (
      <circle
        cx={cx}
        cy={cy}
        r={isSelected ? 6 : 4}
        fill={isSelected ? "rgb(var(--chart-2))" : "rgb(var(--primary))"}
        stroke={isSelected ? "rgb(var(--chart-2))" : "none"}
        strokeWidth={isSelected ? 2 : 0}
        className="cursor-pointer transition-all"
      />
    )
  }

  return (
    <Card className="p-6 bg-card border-border shadow-lg">
      <Tabs defaultValue="line" className="w-full">
        <TabsList className="grid w-full max-w-md grid-cols-2 mb-6 bg-muted">
          <TabsTrigger value="line" className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4" />
            Line Chart
          </TabsTrigger>
          <TabsTrigger value="scatter" className="flex items-center gap-2">
            <ScatterIcon className="w-4 h-4" />
            Scatter Plot
          </TabsTrigger>
        </TabsList>

        <TabsContent value="line" className="mt-0">
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart
                data={chartData}
                onClick={(e) => e?.activeLabel && onSelectCandidate(Number.parseInt(e.activeLabel) - 1)}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgb(var(--border))" opacity={0.3} />
                <XAxis
                  dataKey="index"
                  stroke="rgb(var(--muted-foreground))"
                  label={{
                    value: "Candidate Index",
                    position: "insideBottom",
                    offset: -5,
                    fill: "rgb(var(--muted-foreground))",
                  }}
                  tick={{ fill: "rgb(var(--muted-foreground))" }}
                />
                <YAxis
                  stroke="rgb(var(--muted-foreground))"
                  label={{
                    value: "Probability (%)",
                    angle: -90,
                    position: "insideLeft",
                    fill: "rgb(var(--muted-foreground))",
                  }}
                  tick={{ fill: "rgb(var(--muted-foreground))" }}
                  domain={[0, 100]}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ paddingTop: "20px" }} />
                <ReferenceLine
                  y={90}
                  stroke="rgb(var(--chart-2))"
                  strokeDasharray="3 3"
                  label={{ value: "Confirmed", fill: "rgb(var(--chart-2))", fontSize: 12 }}
                />
                <ReferenceLine
                  y={50}
                  stroke="rgb(var(--chart-4))"
                  strokeDasharray="3 3"
                  label={{ value: "Threshold", fill: "rgb(var(--chart-4))", fontSize: 12 }}
                />
                <ReferenceLine
                  y={10}
                  stroke="rgb(var(--destructive))"
                  strokeDasharray="3 3"
                  label={{ value: "Unlikely", fill: "rgb(var(--destructive))", fontSize: 12 }}
                />
                <Line
                  type="monotone"
                  dataKey="probability"
                  stroke="rgb(var(--primary))"
                  strokeWidth={2}
                  dot={<CustomDot />}
                  activeDot={{ r: 8, fill: "rgb(var(--chart-2))", stroke: "rgb(var(--background))", strokeWidth: 2 }}
                  name="Exoplanet Probability"
                />
              </LineChart>
            </ResponsiveContainer>
          </motion.div>
        </TabsContent>

        <TabsContent value="scatter" className="mt-0">
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart onClick={(e) => e?.activeLabel && onSelectCandidate(Number.parseInt(e.activeLabel) - 1)}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgb(var(--border))" opacity={0.3} />
                <XAxis
                  dataKey="index"
                  stroke="rgb(var(--muted-foreground))"
                  label={{
                    value: "Candidate Index",
                    position: "insideBottom",
                    offset: -5,
                    fill: "rgb(var(--muted-foreground))",
                  }}
                  tick={{ fill: "rgb(var(--muted-foreground))" }}
                />
                <YAxis
                  stroke="rgb(var(--muted-foreground))"
                  label={{
                    value: "Probability (%)",
                    angle: -90,
                    position: "insideLeft",
                    fill: "rgb(var(--muted-foreground))",
                  }}
                  tick={{ fill: "rgb(var(--muted-foreground))" }}
                  domain={[0, 100]}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ paddingTop: "20px" }} />
                <ReferenceLine
                  y={90}
                  stroke="rgb(var(--chart-2))"
                  strokeDasharray="3 3"
                  label={{ value: "Confirmed", fill: "rgb(var(--chart-2))", fontSize: 12 }}
                />
                <ReferenceLine
                  y={50}
                  stroke="rgb(var(--chart-4))"
                  strokeDasharray="3 3"
                  label={{ value: "Threshold", fill: "rgb(var(--chart-4))", fontSize: 12 }}
                />
                <ReferenceLine
                  y={10}
                  stroke="rgb(var(--destructive))"
                  strokeDasharray="3 3"
                  label={{ value: "Unlikely", fill: "rgb(var(--destructive))", fontSize: 12 }}
                />
                <Scatter name="Exoplanet Probability" data={chartData} fill="rgb(var(--primary))" shape="circle" />
              </ScatterChart>
            </ResponsiveContainer>
          </motion.div>
        </TabsContent>
      </Tabs>
    </Card>
  )
}
