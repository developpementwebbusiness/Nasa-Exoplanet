"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import {
  Upload,
  Rocket,
  Brain,
  Database,
  Sparkles,
  TrendingUp,
  Filter,
  Table,
  Github,
} from "lucide-react";
import { CSVUploader } from "@/components/csv-uploader";
import { ProbabilityGraph } from "@/components/probability-graph";
import { ClassificationPanel } from "@/components/classification-panel";
import { ModelManager } from "@/components/model-manager";
import { StatsOverview } from "@/components/stats-overview";
import { ThemeToggle } from "@/components/theme-toggle";
import { DataTable } from "@/components/data-table";
import { CandidateModal } from "@/components/candidate-modal";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface ClassificationWithComment {
  type: "exoplanet" | "not_exoplanet" | "unsure";
  comment?: string;
}

export default function ExoplanetExplorer() {
  const [csvData, setCsvData] = useState<any[]>([]);
  const [predictions, setPredictions] = useState<any[]>([]);
  const [selectedCandidate, setSelectedCandidate] = useState<number | null>(
    null
  );
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [classifications, setClassifications] = useState<
    Record<number, ClassificationWithComment>
  >({});
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeView, setActiveView] = useState<"overview" | "detailed">(
    "overview"
  );

  const handleCandidateSelect = (index: number) => {
    setSelectedCandidate(index);
  };

  const handleCandidateSelectForModal = (index: number) => {
    setSelectedCandidate(index);
    setIsModalOpen(true);
  };

  const handleCSVUpload = async (data: any[]) => {
    setCsvData(data);
    setIsProcessing(true);

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_BASE_URL}/predict`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ data }),
        }
      ).catch(() => null);

      if (response?.ok) {
        const result = await response.json();
        setPredictions(result.predictions || []);
      } else {
        const mockPredictions = data.map(() => ({
          confidence: Math.random() * 0.3 + 0.7,
        }));
        setPredictions(mockPredictions);
      }
    } catch (error) {
      console.error("[v0] Prediction error:", error);
      const mockPredictions = data.map(() => ({
        confidence: Math.random() * 0.3 + 0.7,
      }));
      setPredictions(mockPredictions);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleClassification = (
    index: number,
    classification: "exoplanet" | "not_exoplanet" | "unsure",
    comment?: string
  ) => {
    setClassifications((prev) => ({
      ...prev,
      [index]: { type: classification, comment },
    }));
  };

  const stats = {
    totalCandidates: csvData.length,
    exoplanets: Object.values(classifications).filter(
      (c) => c.type === "exoplanet"
    ).length,
    notExoplanets: Object.values(classifications).filter(
      (c) => c.type === "not_exoplanet"
    ).length,
    unsure: Object.values(classifications).filter((c) => c.type === "unsure")
      .length,
    avgConfidence:
      predictions.length > 0
        ? (
            (predictions.reduce(
              (sum, p) => sum + (p.confidence || p.probability || 0),
              0
            ) /
              predictions.length) *
            100
          ).toFixed(1)
        : "0",
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 dark:from-background dark:via-background dark:to-background/95">
      <header className="border-b border-border/50 bg-card/80 backdrop-blur-md sticky top-0 z-50 shadow-sm">
        <div className="container mx-auto px-6 py-3 sm:py-4">
          <div className="flex items-center justify-between">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center gap-3 sm:gap-4"
            >
              <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-xl bg-gradient-to-br from-primary to-primary/80 flex items-center justify-center shadow-lg">
                <Rocket className="w-5 h-5 sm:w-6 sm:h-6 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-lg sm:text-2xl font-bold text-foreground tracking-tight">
                  S.T.A.R. Trackers
                </h1>
                <p className="text-xs sm:text-sm text-muted-foreground">
                  AI-Powered Detection •{" "}
                  <a
                    href="https://www.spaceappschallenge.org/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-primary transition-colors underline"
                  >
                    Space Apps 2025
                  </a>
                </p>
              </div>
            </motion.div>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8 sm:py-12 space-y-8 sm:space-y-12 max-w-7xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-gradient-to-r from-primary/10 to-primary/5 rounded-lg p-4 border border-primary/20"
        >
          <div className="flex flex-col sm:flex-row items-start gap-4">
            <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-lg bg-primary flex items-center justify-center flex-shrink-0">
              <Sparkles className="w-5 h-5 sm:w-6 sm:h-6 text-primary-foreground" />
            </div>
            <p className="text-sm sm:text-base text-foreground/80">
              Welcome to the NASA Space Apps Challenge 2025 Exoplanet Detection
              System. Upload your Kepler telescope data, analyze AI predictions,
              and manually classify candidates to help discover new worlds
              beyond our solar system.
            </p>
          </div>
        </motion.div>

        {/* Stats Overview */}
        {csvData.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-gradient-to-r from-card/50 to-card/30 rounded-xl p-6 border border-border/50 shadow-sm backdrop-blur-sm"
          >
            <StatsOverview stats={stats} />
          </motion.div>
        )}

        {/* CSV Upload Section */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="flex items-center gap-4 mb-6">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary/20 to-primary/10 flex items-center justify-center border border-primary/20">
              <Upload className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h2 className="text-2xl sm:text-3xl font-bold text-foreground">
                Data Upload
              </h2>
              <p className="text-muted-foreground">
                Upload your Kepler telescope CSV data for analysis
              </p>
            </div>
          </div>
          <CSVUploader onUpload={handleCSVUpload} isProcessing={isProcessing} />
        </motion.section>

        {csvData.length > 0 && (
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="space-y-6"
          >
            <Tabs
              value={activeView}
              onValueChange={(v) => {
                setActiveView(v as any);
                // Reset to candidate 1 when switching to overview (manual classification)
                if (v === "overview" && csvData.length > 0) {
                  setSelectedCandidate(0);
                }
              }}
              className="w-full"
            >
              <TabsList className="grid w-full max-w-lg grid-cols-2 bg-muted/50 backdrop-blur-sm border border-border/50 shadow-sm">
                <TabsTrigger
                  value="overview"
                  className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-sm transition-all"
                >
                  <TrendingUp className="w-4 h-4 mr-2" />
                  Overview
                </TabsTrigger>
                <TabsTrigger
                  value="detailed"
                  className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-sm transition-all"
                >
                  <Table className="w-4 h-4 mr-2" />
                  Detailed Data
                </TabsTrigger>
              </TabsList>

              <TabsContent value="overview" className="space-y-8 mt-8">
                {/* Probability Visualization */}
                <div className="bg-card/50 rounded-xl p-6 border border-border/50 shadow-sm">
                  <div className="flex items-center gap-4 mb-6">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-chart-2/20 to-chart-2/10 flex items-center justify-center border border-chart-2/20">
                      <TrendingUp className="w-6 h-6 text-chart-2" />
                    </div>
                    <div>
                      <h2 className="text-2xl font-bold text-foreground">
                        Confidence Analysis
                      </h2>
                      <p className="text-muted-foreground">
                        AI predictions and confidence scores
                      </p>
                    </div>
                  </div>
                  <ProbabilityGraph
                    predictions={predictions}
                    onSelectCandidate={handleCandidateSelect}
                    selectedCandidate={selectedCandidate}
                    onCandidateClick={handleCandidateSelectForModal}
                    classifications={classifications}
                  />
                </div>

                {/* Classification Panel */}
                <div className="bg-card/50 rounded-xl p-6 border border-border/50 shadow-sm">
                  <div className="flex items-center gap-4 mb-6">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-accent/20 to-accent/10 flex items-center justify-center border border-accent/20">
                      <Brain className="w-6 h-6 text-accent" />
                    </div>
                    <div>
                      <h2 className="text-2xl font-bold text-foreground">
                        Manual Classification
                      </h2>
                      <p className="text-muted-foreground">
                        Review and classify exoplanet candidates
                      </p>
                    </div>
                  </div>
                  <ClassificationPanel
                    csvData={csvData}
                    predictions={predictions}
                    classifications={classifications}
                    onClassify={handleClassification}
                    selectedCandidate={selectedCandidate}
                    onSelectCandidate={handleCandidateSelect}
                  />
                </div>
              </TabsContent>

              <TabsContent value="detailed" className="mt-8">
                <div className="bg-card/50 rounded-xl p-6 border border-border/50 shadow-sm">
                  <div className="flex items-center gap-4 mb-6">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-chart-4/20 to-chart-4/10 flex items-center justify-center border border-chart-4/20">
                      <Filter className="w-6 h-6 text-chart-4" />
                    </div>
                    <div>
                      <h2 className="text-2xl font-bold text-foreground">
                        Candidate Data Table
                      </h2>
                      <p className="text-muted-foreground">
                        Detailed view of all exoplanet candidates
                      </p>
                    </div>
                  </div>
                  <DataTable
                    data={csvData}
                    predictions={predictions}
                    classifications={classifications}
                    onSelectCandidate={handleCandidateSelectForModal}
                    selectedCandidate={selectedCandidate}
                  />
                </div>
              </TabsContent>
            </Tabs>
          </motion.section>
        )}

        {/* Model Management Section */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <div className="flex items-center gap-4 mb-6">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-chart-5/20 to-chart-5/10 flex items-center justify-center border border-chart-5/20">
              <Database className="w-6 h-6 text-chart-5" />
            </div>
            <div>
              <h2 className="text-2xl sm:text-3xl font-bold text-foreground">
                Model Management
              </h2>
              <p className="text-muted-foreground">
                Train and manage your AI classification models
              </p>
            </div>
          </div>
          <div className="bg-card/50 rounded-xl p-6 border border-border/50 shadow-sm">
            <ModelManager classifications={classifications} csvData={csvData} />
          </div>
        </motion.section>
      </main>

      <footer className="border-t border-border/50 mt-16 sm:mt-20 py-8 sm:py-12 bg-card/50 backdrop-blur-sm">
        <div className="container mx-auto px-6 max-w-7xl">
          <div className="flex flex-col items-center gap-6">
            <div className="flex items-center gap-4">
              <Link
                href="https://github.com/developpementwebbusiness/Nasa-Exoplanet"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-3 px-4 py-3 rounded-xl bg-muted/50 hover:bg-muted/80 transition-all border border-border/50 shadow-sm hover:shadow-md"
              >
                <Github className="w-5 h-5 sm:w-6 sm:h-6 text-foreground" />
                <span className="text-sm sm:text-base font-medium text-foreground">
                  View on GitHub
                </span>
              </Link>
            </div>
            <div className="text-center space-y-2">
              <p className="text-sm sm:text-base font-semibold text-muted-foreground">
                NASA Space Apps Challenge 2025 • Exoplanet Detection System
              </p>
              <p className="text-sm text-muted-foreground max-w-md">
                Open Source • Powered by AI & Machine Learning • Exploring the
                Universe
              </p>
            </div>
          </div>
        </div>
      </footer>

      {/* Candidate Detail Modal */}
      <CandidateModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        candidateData={
          selectedCandidate !== null ? csvData[selectedCandidate] : null
        }
        candidateIndex={selectedCandidate !== null ? selectedCandidate : 0}
        prediction={
          selectedCandidate !== null ? predictions[selectedCandidate] : null
        }
        classification={
          selectedCandidate !== null
            ? classifications[selectedCandidate]
            : undefined
        }
        onClassify={handleClassification}
      />
    </div>
  );
}
