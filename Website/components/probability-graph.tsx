"use client";

import { useMemo, useState, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
  Brush,
} from "recharts";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Download,
  Filter,
  TrendingUp,
  BarChart3,
  RefreshCw,
  ZoomIn,
} from "lucide-react";

interface ConfidenceGraphProps {
  predictions: any[];
  onSelectCandidate: (index: number) => void;
  selectedCandidate: number | null;
  onCandidateClick?: (index: number) => void;
  classifications?: Record<number, { type: string; comment?: string }>;
}

export function ProbabilityGraph({
  predictions,
  onSelectCandidate,
  selectedCandidate,
  onCandidateClick,
  classifications = {},
}: ConfidenceGraphProps) {
  // All hooks MUST come before any conditional returns
  const [filter, setFilter] = useState<"all" | "high" | "medium" | "low">(
    "all"
  );
  const [showGrid, setShowGrid] = useState(true);
  const [lineType, setLineType] = useState<"monotone" | "linear" | "step">(
    "monotone"
  );
  const [strokeWidth, setStrokeWidth] = useState(2);
  const [brushIndexes, setBrushIndexes] = useState<{
    startIndex: number;
    endIndex: number;
  } | null>(null);
  const [isMouseDown, setIsMouseDown] = useState(false);

  const allChartData = useMemo(
    () =>
      predictions.map((pred, index) => ({
        index: index + 1,
        confidence: (pred.confidence || pred.probability || 0) * 100,
        originalIndex: index,
      })),
    [predictions]
  );

  const chartData = useMemo(() => {
    let filtered;
    switch (filter) {
      case "high":
        filtered = allChartData.filter((d) => d.confidence >= 70);
        break;
      case "medium":
        filtered = allChartData.filter(
          (d) => d.confidence >= 30 && d.confidence < 70
        );
        break;
      case "low":
        filtered = allChartData.filter((d) => d.confidence < 30);
        break;
      default:
        filtered = allChartData;
    }
    return filtered;
  }, [allChartData, filter]);

  const stats = useMemo(() => {
    if (chartData.length === 0) {
      return {
        max: "0",
        min: "0",
        avg: "0",
        total: 0,
        high: 0,
        medium: 0,
        low: 0,
      };
    }
    const confidences = chartData.map((d) => d.confidence);
    const max = Math.max(...confidences);
    const min = Math.min(...confidences);
    const sum = confidences.reduce((a, b) => a + b, 0);
    return {
      max: max.toFixed(1),
      min: min.toFixed(1),
      avg: (sum / confidences.length).toFixed(1),
      total: chartData.length,
      high: chartData.filter((d) => d.confidence >= 70).length,
      medium: chartData.filter((d) => d.confidence >= 30 && d.confidence < 70)
        .length,
      low: chartData.filter((d) => d.confidence < 30).length,
    };
  }, [chartData]);

  const CustomTooltip = useCallback(({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const conf = payload[0].value;
      const category = conf >= 70 ? "High" : conf >= 30 ? "Medium" : "Low";
      const color = conf >= 70 ? "#22c55e" : conf >= 30 ? "#f59e0b" : "#ef4444";

      return (
        <div className="bg-card border-2 border-primary rounded-lg p-4 shadow-xl">
          <p className="text-sm font-bold text-foreground mb-2">
            Candidate #{data.index}
          </p>
          <div className="flex items-center gap-2 mb-2">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: color }}
            />
            <p className="text-xl font-bold text-primary">{conf.toFixed(2)}%</p>
          </div>
          <Badge style={{ backgroundColor: color, color: "white" }}>
            {category} Confidence
          </Badge>
        </div>
      );
    }
    return null;
  }, []);

  const downloadCSV = useCallback(() => {
    const csv = [
      ["Candidate Index", "Confidence %", "Classification", "Notes"],
      ...chartData.map((d) => {
        const classification = classifications[d.originalIndex];
        return [
          d.index,
          d.confidence.toFixed(2),
          classification
            ? classification.type.replace(/_/g, " ")
            : "Not classified",
          classification?.comment || "",
        ];
      }),
    ]
      .map((row) => row.map((cell) => `"${cell}"`).join(","))
      .join("\n");

    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `exoplanet-confidence-${filter}-${
      new Date().toISOString().split("T")[0]
    }.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [chartData, filter, classifications]);

  const resetView = useCallback(() => {
    setFilter("all");
    setShowGrid(true);
    setLineType("monotone");
    setStrokeWidth(2);
    setBrushIndexes(null);
  }, []);

  const handleBrushChange = useCallback(
    (brushData: any) => {
      if (
        !isMouseDown &&
        brushData &&
        brushData.startIndex !== undefined &&
        brushData.endIndex !== undefined
      ) {
        // Only update when mouse is released
        setBrushIndexes({
          startIndex: brushData.startIndex,
          endIndex: brushData.endIndex,
        });
      }
    },
    [isMouseDown]
  );

  // NOW check if empty AFTER all hooks
  if (predictions.length === 0) {
    return (
      <Card className="p-8 text-center bg-card border-border">
        <p className="text-muted-foreground">
          Upload CSV data to see confidence analysis
        </p>
      </Card>
    );
  }

  return (
    <Card className="p-6 bg-card border-border shadow-lg">
      {/* Header with Stats */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-foreground">
            AI Prediction Confidence
          </h3>
          <p className="text-xs text-muted-foreground mt-1">
            {stats.total} candidates • Avg: {stats.avg}% • Max: {stats.max}% •
            Min: {stats.min}%
          </p>
        </div>
        <div className="flex gap-2">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                onClick={downloadCSV}
                className="h-8"
              >
                <Download className="w-4 h-4 mr-1" />
                Export
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Download confidence data as CSV file</p>
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                onClick={resetView}
                className="h-8"
              >
                <RefreshCw className="w-4 h-4 mr-1" />
                Reset
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Reset all filters and settings to default</p>
            </TooltipContent>
          </Tooltip>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-2 mb-4">
        <div className="bg-muted rounded-lg p-2">
          <p className="text-xs text-muted-foreground">Total</p>
          <p className="text-lg font-bold text-foreground">{stats.total}</p>
        </div>
        <div className="bg-green-500/10 rounded-lg p-2">
          <p className="text-xs text-green-600 dark:text-green-400">
            High (&gt;70%)
          </p>
          <p className="text-lg font-bold text-green-600 dark:text-green-400">
            {stats.high}
          </p>
        </div>
        <div className="bg-orange-500/10 rounded-lg p-2">
          <p className="text-xs text-orange-600 dark:text-orange-400">
            Medium (30-70%)
          </p>
          <p className="text-lg font-bold text-orange-600 dark:text-orange-400">
            {stats.medium}
          </p>
        </div>
        <div className="bg-red-500/10 rounded-lg p-2">
          <p className="text-xs text-red-600 dark:text-red-400">
            Low (&lt;30%)
          </p>
          <p className="text-lg font-bold text-red-600 dark:text-red-400">
            {stats.low}
          </p>
        </div>
        <div className="bg-primary/10 rounded-lg p-2">
          <p className="text-xs text-primary">Average</p>
          <p className="text-lg font-bold text-primary">{stats.avg}%</p>
        </div>
      </div>

      {/* Filter Controls */}
      <div className="flex flex-wrap gap-2 mb-4">
        <div className="flex gap-1">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant={filter === "all" ? "default" : "outline"}
                size="sm"
                onClick={() => setFilter("all")}
                className="h-8"
              >
                <Filter className="w-3 h-3 mr-1" />
                All
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Show all candidates</p>
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant={filter === "high" ? "default" : "outline"}
                size="sm"
                onClick={() => setFilter("high")}
                className="h-8"
              >
                High
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Show only high confidence candidates (&gt;70%)</p>
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant={filter === "medium" ? "default" : "outline"}
                size="sm"
                onClick={() => setFilter("medium")}
                className="h-8"
              >
                Medium
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Show only medium confidence candidates (30-70%)</p>
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant={filter === "low" ? "default" : "outline"}
                size="sm"
                onClick={() => setFilter("low")}
                className="h-8"
              >
                Low
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Show only low confidence candidates (&lt;30%)</p>
            </TooltipContent>
          </Tooltip>
        </div>

        <div className="flex gap-1 ml-auto">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowGrid(!showGrid)}
                className="h-8"
              >
                <BarChart3 className="w-3 h-3 mr-1" />
                Grid
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Toggle grid lines on/off</p>
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                onClick={() =>
                  setLineType((prev) =>
                    prev === "monotone"
                      ? "linear"
                      : prev === "linear"
                      ? "step"
                      : "monotone"
                  )
                }
                className="h-8"
              >
                <TrendingUp className="w-3 h-3 mr-1" />
                {lineType}
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>
                Change line style: monotone (smooth), linear (straight), or step
              </p>
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                onClick={() =>
                  setStrokeWidth((prev) =>
                    prev === 2 ? 3 : prev === 3 ? 4 : 2
                  )
                }
                className="h-8"
              >
                <ZoomIn className="w-3 h-3 mr-1" />
                {strokeWidth}px
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Adjust line thickness: 2px, 3px, or 4px</p>
            </TooltipContent>
          </Tooltip>
        </div>
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={550}>
        <LineChart
          data={chartData}
          onClick={(e) => {
            if (e?.activePayload?.[0]?.payload) {
              const clickedData = e.activePayload[0].payload;
              onSelectCandidate(clickedData.originalIndex);
            }
          }}
          margin={{ top: 10, right: 30, left: 0, bottom: 80 }}
        >
          <defs>
            <linearGradient id="lineGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.8} />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.2} />
            </linearGradient>
          </defs>
          {showGrid && (
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#374151"
              opacity={0.3}
            />
          )}
          <XAxis
            dataKey="index"
            stroke="#9CA3AF"
            tick={{ fill: "#9CA3AF", fontSize: 11 }}
            tickLine={false}
            label={{
              value: "Candidate Index",
              position: "insideBottom",
              offset: -10,
              style: { fill: "#9CA3AF", fontSize: 12 },
            }}
          />
          <YAxis
            stroke="#9CA3AF"
            tick={{ fill: "#9CA3AF", fontSize: 11 }}
            tickLine={false}
            domain={[0, 100]}
            label={{
              value: "Confidence %",
              angle: -90,
              position: "insideLeft",
              style: { fill: "#9CA3AF", fontSize: 12 },
            }}
          />
          <RechartsTooltip
            content={<CustomTooltip />}
            cursor={{ stroke: "#3b82f6", strokeWidth: 1 }}
          />
          <Legend
            wrapperStyle={{
              paddingTop: "15px",
              paddingBottom: "60px",
              fontSize: "16px",
            }}
            iconType="line"
            formatter={() => (
              <span className="text-base sm:text-lg md:text-xl font-bold">
                AI Confidence Score
              </span>
            )}
            verticalAlign="top"
          />

          {/* Reference Lines */}
          <ReferenceLine
            y={70}
            stroke="#22c55e"
            strokeDasharray="5 5"
            strokeWidth={2}
            label={{
              value: "High (70%)",
              position: "right",
              fill: "#22c55e",
              fontSize: 11,
            }}
          />
          <ReferenceLine
            y={30}
            stroke="#ef4444"
            strokeDasharray="5 5"
            strokeWidth={2}
            label={{
              value: "Low (30%)",
              position: "right",
              fill: "#ef4444",
              fontSize: 11,
            }}
          />

          {/* Main Line */}
          <Line
            type={lineType}
            dataKey="confidence"
            stroke="#3b82f6"
            strokeWidth={strokeWidth}
            dot={{ fill: "#3b82f6", r: 2 }}
            activeDot={{
              r: 6,
              fill: "#3b82f6",
              stroke: "#fff",
              strokeWidth: 2,
              onClick: (e: any) => {
                if (onCandidateClick && e.payload) {
                  onCandidateClick(e.payload.originalIndex);
                }
              },
              cursor: "pointer",
            }}
            isAnimationActive={false}
            onClick={(data: any) => {
              if (onCandidateClick && data) {
                onCandidateClick(data.originalIndex);
              }
            }}
            cursor="pointer"
          />

          {/* Zoom/Pan Brush - Below Legend, Shows All Data by Default */}
          <Brush
            dataKey="index"
            height={35}
            stroke="#3b82f6"
            fill="rgba(59, 130, 246, 0.1)"
            travellerWidth={12}
            y={480}
            onChange={handleBrushChange}
            startIndex={brushIndexes?.startIndex}
            endIndex={brushIndexes?.endIndex}
            onMouseDown={() => setIsMouseDown(true)}
            onMouseUp={() => setIsMouseDown(false)}
          />
        </LineChart>
      </ResponsiveContainer>

      {/* Legend Info */}
      <div className="mt-4 flex flex-wrap gap-4 text-xs text-muted-foreground">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-green-500" />
          <span>High Confidence (&gt;70%): Likely exoplanet</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-orange-500" />
          <span>Medium (30-70%): Uncertain</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <span>Low (&lt;30%): Unlikely exoplanet</span>
        </div>
      </div>
    </Card>
  );
}
