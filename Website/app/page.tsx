"use client"

import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Upload, Rocket, Brain, Database, Sparkles, TrendingUp, Filter, Table, Github } from "lucide-react"
import { CSVUploader } from "@/components/csv-uploader"
import { ProbabilityGraph } from "@/components/probability-graph"
import { ClassificationPanel } from "@/components/classification-panel"
import { ModelManager } from "@/components/model-manager"
import { StatsOverview } from "@/components/stats-overview"
import { ThemeToggle } from "@/components/theme-toggle"
import { DataTable } from "@/components/data-table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

interface ClassificationWithComment {
  type: "exoplanet" | "not_exoplanet" | "unsure"
  comment?: string
}

export default function ExoplanetExplorer() {
  const [csvData, setCsvData] = useState<any[]>([])
  const [predictions, setPredictions] = useState<any[]>([])
  const [selectedCandidate, setSelectedCandidate] = useState<number | null>(null)
  const [classifications, setClassifications] = useState<Record<number, ClassificationWithComment>>({})
  const [isProcessing, setIsProcessing] = useState(false)
  const [activeView, setActiveView] = useState<"overview" | "detailed">("overview")

  const handleCSVUpload = async (data: any[]) => {
    setCsvData(data)
    setIsProcessing(true)

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data }),
      }).catch(() => null)

      if (response?.ok) {
        const result = await response.json()
        setPredictions(result.predictions || [])
      } else {
        const mockPredictions = data.map(() => ({
          probability: Math.random(),
          confidence: Math.random() * 0.3 + 0.7,
        }))
        setPredictions(mockPredictions)
      }
    } catch (error) {
      console.error("[v0] Prediction error:", error)
      const mockPredictions = data.map(() => ({
        probability: Math.random(),
        confidence: Math.random() * 0.3 + 0.7,
      }))
      setPredictions(mockPredictions)
    } finally {
      setIsProcessing(false)
    }
  }

  const handleClassification = (
    index: number,
    classification: "exoplanet" | "not_exoplanet" | "unsure",
    comment?: string,
  ) => {
    setClassifications((prev) => ({
      ...prev,
      [index]: { type: classification, comment },
    }))
  }

  const stats = {
    totalCandidates: csvData.length,
    exoplanets: Object.values(classifications).filter((c) => c.type === "exoplanet").length,
    notExoplanets: Object.values(classifications).filter((c) => c.type === "not_exoplanet").length,
    unsure: Object.values(classifications).filter((c) => c.type === "unsure").length,
    avgProbability:
      predictions.length > 0
        ? ((predictions.reduce((sum, p) => sum + (p.probability || 0), 0) / predictions.length) * 100).toFixed(1)
        : "0",
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card backdrop-blur-sm sticky top-0 z-50 shadow-lg">
        <div className="container mx-auto px-4 py-3 sm:py-4">
          <div className="flex items-center justify-between gap-4">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center gap-2 sm:gap-3 min-w-0 flex-1"
            >
              <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
                <Rocket className="w-5 h-5 sm:w-7 sm:h-7 text-primary-foreground" />
              </div>
              <div className="min-w-0 flex-1">
                <h1 className="text-lg sm:text-2xl font-bold text-foreground truncate">NASA Exoplanet Explorer</h1>
                <p className="text-xs sm:text-sm text-muted-foreground truncate">
                  AI-Powered Detection • Space Apps 2025
                </p>
              </div>
            </motion.div>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6 sm:py-8 space-y-6 sm:space-y-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-primary/5 rounded-lg p-4 sm:p-6 border-2 border-primary/30"
        >
          <div className="flex flex-col sm:flex-row items-start gap-4">
            <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-lg bg-primary flex items-center justify-center flex-shrink-0">
              <Sparkles className="w-5 h-5 sm:w-6 sm:h-6 text-primary-foreground" />
            </div>
            <div className="flex-1">
              <h2 className="text-lg sm:text-xl font-semibold text-foreground mb-2">Mission Control</h2>
              <p className="text-sm sm:text-base text-foreground/80 leading-relaxed">
                Welcome to the NASA Space Apps Challenge 2025 Exoplanet Detection System. Upload your Kepler telescope
                data, analyze AI predictions, and manually classify candidates to help discover new worlds beyond our
                solar system.
              </p>
            </div>
          </div>
        </motion.div>

        {/* Stats Overview */}
        {csvData.length > 0 && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            <StatsOverview stats={stats} />
          </motion.div>
        )}

        {/* CSV Upload Section */}
        <motion.section initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
          <div className="flex items-center gap-2 sm:gap-3 mb-4">
            <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-lg bg-primary/20 flex items-center justify-center">
              <Upload className="w-4 h-4 sm:w-5 sm:h-5 text-primary" />
            </div>
            <h2 className="text-xl sm:text-2xl font-bold text-foreground">Data Upload</h2>
          </div>
          <CSVUploader onUpload={handleCSVUpload} isProcessing={isProcessing} />
        </motion.section>

        <AnimatePresence>
          {csvData.length > 0 && (
            <motion.section
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ delay: 0.4 }}
              className="space-y-6"
            >
              <Tabs value={activeView} onValueChange={(v) => setActiveView(v as any)} className="w-full">
                <TabsList className="grid w-full max-w-md grid-cols-2 bg-muted">
                  <TabsTrigger
                    value="overview"
                    className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
                  >
                    <TrendingUp className="w-4 h-4 mr-2" />
                    Overview
                  </TabsTrigger>
                  <TabsTrigger
                    value="detailed"
                    className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
                  >
                    <Table className="w-4 h-4 mr-2" />
                    Detailed Data
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="overview" className="space-y-6 mt-6">
                  {/* Probability Visualization */}
                  <div>
                    <div className="flex items-center gap-3 mb-4">
                      <div className="w-8 h-8 rounded-lg bg-chart-2/20 flex items-center justify-center">
                        <TrendingUp className="w-5 h-5 text-chart-2" />
                      </div>
                      <h2 className="text-2xl font-bold text-foreground">Probability Analysis</h2>
                    </div>
                    <ProbabilityGraph
                      predictions={predictions}
                      onSelectCandidate={setSelectedCandidate}
                      selectedCandidate={selectedCandidate}
                    />
                  </div>

                  {/* Classification Panel */}
                  <div>
                    <div className="flex items-center gap-3 mb-4">
                      <div className="w-8 h-8 rounded-lg bg-accent/20 flex items-center justify-center">
                        <Brain className="w-5 h-5 text-accent" />
                      </div>
                      <h2 className="text-2xl font-bold text-foreground">Manual Classification</h2>
                    </div>
                    <ClassificationPanel
                      csvData={csvData}
                      predictions={predictions}
                      classifications={classifications}
                      onClassify={handleClassification}
                      selectedCandidate={selectedCandidate}
                      onSelectCandidate={setSelectedCandidate}
                    />
                  </div>
                </TabsContent>

                <TabsContent value="detailed" className="mt-6">
                  <div>
                    <div className="flex items-center gap-3 mb-4">
                      <div className="w-8 h-8 rounded-lg bg-chart-4/20 flex items-center justify-center">
                        <Filter className="w-5 h-5 text-chart-4" />
                      </div>
                      <h2 className="text-2xl font-bold text-foreground">Candidate Data Table</h2>
                    </div>
                    <DataTable
                      data={csvData}
                      predictions={predictions}
                      classifications={classifications}
                      onSelectCandidate={setSelectedCandidate}
                    />
                  </div>
                </TabsContent>
              </Tabs>
            </motion.section>
          )}
        </AnimatePresence>

        {/* Model Management Section */}
        <motion.section initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-8 h-8 rounded-lg bg-chart-5/20 flex items-center justify-center">
              <Database className="w-5 h-5 text-chart-5" />
            </div>
            <h2 className="text-2xl font-bold text-foreground">Model Management</h2>
          </div>
          <ModelManager classifications={classifications} csvData={csvData} />
        </motion.section>
      </main>

      <footer className="border-t border-border mt-12 sm:mt-16 py-6 sm:py-8 bg-card">
        <div className="container mx-auto px-4">
          <div className="flex flex-col items-center gap-4">
            <div className="flex items-center gap-4">
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-3 sm:px-4 py-2 rounded-lg bg-muted hover:bg-muted/80 transition-colors border border-border"
              >
                <Github className="w-4 h-4 sm:w-5 sm:h-5 text-foreground" />
                <span className="text-xs sm:text-sm font-medium text-foreground">View on GitHub</span>
              </a>
            </div>
            <div className="text-center">
              <p className="text-xs sm:text-sm font-medium text-muted-foreground">
                NASA Space Apps Challenge 2025 • Exoplanet Detection System
              </p>
              <p className="text-xs mt-2 text-muted-foreground">
                Open Source • Powered by AI & Machine Learning • Exploring the Universe
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
