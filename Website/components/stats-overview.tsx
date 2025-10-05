"use client";

import { Card } from "@/components/ui/card";
import {
  Database,
  CheckCircle2,
  XCircle,
  HelpCircle,
  TrendingUp,
} from "lucide-react";

interface StatsOverviewProps {
  stats: {
    totalCandidates: number;
    exoplanets: number;
    notExoplanets: number;
    unsure: number;
    avgConfidence: string;
  };
}

export function StatsOverview({ stats }: StatsOverviewProps) {
  const statCards = [
    {
      label: "Total Candidates",
      value: stats.totalCandidates,
      icon: Database,
      color: "text-chart-1",
      bgColor: "bg-chart-1/10",
    },
    {
      label: "Confirmed",
      value: stats.exoplanets,
      icon: CheckCircle2,
      color: "text-chart-2",
      bgColor: "bg-chart-2/10",
    },
    {
      label: "Rejected",
      value: stats.notExoplanets,
      icon: XCircle,
      color: "text-chart-3",
      bgColor: "bg-chart-3/10",
    },
    {
      label: "Pending",
      value: stats.unsure,
      icon: HelpCircle,
      color: "text-chart-5",
      bgColor: "bg-chart-5/10",
    },
    {
      label: "Avg Confidence",
      value: `${stats.avgConfidence}%`,
      icon: TrendingUp,
      color: "text-chart-4",
      bgColor: "bg-chart-4/10",
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
      {statCards.map((stat) => (
        <Card
          key={stat.label}
          className="p-4 bg-card border-border transition-all hover:border-primary/50 hover:shadow-md"
        >
          <div className="flex items-center justify-between gap-3 mb-3">
            <div
              className={`w-10 h-10 rounded-lg ${stat.bgColor} flex items-center justify-center flex-shrink-0`}
            >
              <stat.icon className={`w-5 h-5 ${stat.color}`} />
            </div>
          </div>
          <div>
            <p className="text-2xl font-bold text-foreground mb-1">
              {stat.value}
            </p>
            <p className="text-xs text-muted-foreground font-medium">
              {stat.label}
            </p>
          </div>
        </Card>
      ))}
    </div>
  );
}
