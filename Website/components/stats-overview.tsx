"use client"

import { motion } from "framer-motion"
import { Card } from "@/components/ui/card"
import { Database, CheckCircle2, XCircle, HelpCircle, TrendingUp, Info } from "lucide-react"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

interface StatsOverviewProps {
  stats: {
    totalCandidates: number
    exoplanets: number
    notExoplanets: number
    unsure: number
    avgProbability: string
  }
}

export function StatsOverview({ stats }: StatsOverviewProps) {
  const statCards = [
    {
      label: "Total Candidates",
      value: stats.totalCandidates,
      icon: Database,
      color: "text-primary",
      bgColor: "bg-primary/10",
      tooltip: "Total number of exoplanet candidates loaded from your CSV file",
    },
    {
      label: "Exoplanets",
      value: stats.exoplanets,
      icon: CheckCircle2,
      color: "text-chart-4",
      bgColor: "bg-chart-4/10",
      tooltip: "Candidates you've manually classified as confirmed exoplanets",
    },
    {
      label: "Not Exoplanets",
      value: stats.notExoplanets,
      icon: XCircle,
      color: "text-chart-2",
      bgColor: "bg-chart-2/10",
      tooltip: "Candidates you've classified as false positives or non-exoplanets",
    },
    {
      label: "Unsure",
      value: stats.unsure,
      icon: HelpCircle,
      color: "text-chart-5",
      bgColor: "bg-chart-5/10",
      tooltip: "Candidates marked as uncertain and requiring further analysis",
    },
    {
      label: "Avg Probability",
      value: `${stats.avgProbability}%`,
      icon: TrendingUp,
      color: "text-chart-3",
      bgColor: "bg-chart-3/10",
      tooltip: "Average AI-predicted probability across all candidates (higher = more likely to be an exoplanet)",
    },
  ]

  return (
    <TooltipProvider>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {statCards.map((stat, index) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <Card className="p-4 hover:shadow-lg transition-all border-2 border-primary/20 hover:border-primary/40 bg-card">
              <div className="flex items-start justify-between gap-2 mb-2">
                <div className={`w-10 h-10 rounded-lg ${stat.bgColor} flex items-center justify-center flex-shrink-0`}>
                  <stat.icon className={`w-5 h-5 ${stat.color}`} />
                </div>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Info className="w-4 h-4 text-muted-foreground cursor-help hover:text-primary transition-colors" />
                  </TooltipTrigger>
                  <TooltipContent side="top" className="max-w-xs bg-primary text-primary-foreground">
                    <p className="text-sm">{stat.tooltip}</p>
                  </TooltipContent>
                </Tooltip>
              </div>
              <div className="min-w-0">
                <motion.p
                  className="text-2xl font-bold text-foreground"
                  initial={{ scale: 0.5 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: index * 0.1 + 0.2, type: "spring" }}
                >
                  {stat.value}
                </motion.p>
                <p className="text-xs text-muted-foreground truncate font-medium">{stat.label}</p>
              </div>
            </Card>
          </motion.div>
        ))}
      </div>
    </TooltipProvider>
  )
}
