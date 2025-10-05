"use client"

import { motion, AnimatePresence } from "framer-motion"
import { CheckCircle2, XCircle, HelpCircle, ChevronLeft, ChevronRight, MessageSquare } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Textarea } from "@/components/ui/textarea"
import { useState } from "react"

interface ClassificationWithComment {
  type: "exoplanet" | "not_exoplanet" | "unsure"
  comment?: string
}

interface ClassificationPanelProps {
  csvData: any[]
  predictions: any[]
  classifications: Record<number, ClassificationWithComment>
  onClassify: (index: number, classification: "exoplanet" | "not_exoplanet" | "unsure", comment?: string) => void
  selectedCandidate: number | null
  onSelectCandidate: (index: number) => void
}

export function ClassificationPanel({
  csvData,
  predictions,
  classifications,
  onClassify,
  selectedCandidate,
  onSelectCandidate,
}: ClassificationPanelProps) {
  const currentIndex = selectedCandidate ?? 0
  const currentData = csvData[currentIndex]
  const currentPrediction = predictions[currentIndex]
  const currentClassification = classifications[currentIndex]

  const [comment, setComment] = useState(currentClassification?.comment || "")

  const handlePrevious = () => {
    if (currentIndex > 0) {
      const newIndex = currentIndex - 1
      onSelectCandidate(newIndex)
      setComment(classifications[newIndex]?.comment || "")
    }
  }

  const handleNext = () => {
    if (currentIndex < csvData.length - 1) {
      const newIndex = currentIndex + 1
      onSelectCandidate(newIndex)
      setComment(classifications[newIndex]?.comment || "")
    }
  }

  const handleClassify = (type: "exoplanet" | "not_exoplanet" | "unsure") => {
    onClassify(currentIndex, type, comment)
  }

  if (csvData.length === 0) {
    return (
      <Card className="p-8 text-center bg-card border-2 border-border">
        <p className="text-muted-foreground">Upload CSV data to start classification</p>
      </Card>
    )
  }

  const probability = currentPrediction ? (currentPrediction.probability * 100).toFixed(2) : "N/A"

  return (
    <Card className="p-4 sm:p-6 bg-card border-2 border-primary/30 shadow-lg">
      <div className="space-y-4 sm:space-y-6">
        {/* Navigation */}
        <div className="flex items-center justify-between gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handlePrevious}
            disabled={currentIndex === 0}
            className="border-primary/30 bg-background text-foreground hover:bg-primary/10 text-xs sm:text-sm"
          >
            <ChevronLeft className="w-3 h-3 sm:w-4 sm:h-4 sm:mr-1" />
            <span className="hidden sm:inline">Previous</span>
          </Button>

          <div className="text-center">
            <p className="text-xs sm:text-sm text-muted-foreground">Candidate</p>
            <p className="text-xl sm:text-2xl font-bold text-primary">
              {currentIndex + 1} / {csvData.length}
            </p>
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={handleNext}
            disabled={currentIndex === csvData.length - 1}
            className="border-primary/30 bg-background text-foreground hover:bg-primary/10 text-xs sm:text-sm"
          >
            <span className="hidden sm:inline">Next</span>
            <ChevronRight className="w-3 h-3 sm:w-4 sm:h-4 sm:ml-1" />
          </Button>
        </div>

        {/* AI Prediction */}
        <div className="bg-primary/10 border-2 border-primary/30 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm font-semibold text-foreground">AI Prediction Confidence</p>
            <Badge
              className={
                Number.parseFloat(probability) > 50
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-muted-foreground"
              }
            >
              {probability}%
            </Badge>
          </div>
          <div className="w-full bg-muted rounded-full h-3 border border-border">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${probability}%` }}
              transition={{ duration: 0.8, ease: "easeOut" }}
              className="bg-gradient-to-r from-primary to-accent h-3 rounded-full"
            />
          </div>
        </div>

        {/* Data Preview */}
        <div className="bg-card border-2 border-border rounded-lg p-4">
          <p className="text-sm font-semibold text-foreground mb-3">Candidate Data Preview</p>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3 max-h-64 overflow-y-auto">
            {currentData &&
              Object.entries(currentData)
                .slice(0, 15)
                .map(([key, value]: [string, any]) => (
                  <div key={key} className="text-xs bg-muted/50 p-2 rounded border border-border">
                    <p className="text-muted-foreground truncate font-medium">{key}</p>
                    <p className="font-mono text-foreground font-semibold">
                      {typeof value === "number" ? value.toFixed(4) : String(value)}
                    </p>
                  </div>
                ))}
          </div>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-semibold text-foreground flex items-center gap-2">
            <MessageSquare className="w-4 h-4" />
            Optional Comment
          </label>
          <Textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="Add notes about this classification..."
            className="min-h-[80px] bg-background text-foreground border-2 border-border"
          />
        </div>

        {/* Classification Buttons */}
        <AnimatePresence mode="wait">
          <motion.div
            key={currentIndex}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
            className="space-y-3"
          >
            <div className="grid grid-cols-3 gap-2 sm:gap-3">
              <Button
                variant={currentClassification?.type === "exoplanet" ? "default" : "outline"}
                className={`h-auto py-3 sm:py-4 flex-col gap-1 sm:gap-2 border-2 text-xs sm:text-sm ${
                  currentClassification?.type === "exoplanet"
                    ? "bg-chart-4 text-white border-chart-4 hover:bg-chart-4/90"
                    : "border-border bg-background text-foreground hover:bg-muted"
                }`}
                onClick={() => handleClassify("exoplanet")}
              >
                <CheckCircle2 className="w-5 h-5 sm:w-6 sm:h-6" />
                <span className="font-semibold">Exoplanet</span>
              </Button>

              <Button
                variant={currentClassification?.type === "not_exoplanet" ? "default" : "outline"}
                className={`h-auto py-3 sm:py-4 flex-col gap-1 sm:gap-2 border-2 text-xs sm:text-sm ${
                  currentClassification?.type === "not_exoplanet"
                    ? "bg-chart-2 text-white border-chart-2 hover:bg-chart-2/90"
                    : "border-border bg-background text-foreground hover:bg-muted"
                }`}
                onClick={() => handleClassify("not_exoplanet")}
              >
                <XCircle className="w-5 h-5 sm:w-6 sm:h-6" />
                <span className="font-semibold">Not Exoplanet</span>
              </Button>

              <Button
                variant={currentClassification?.type === "unsure" ? "default" : "outline"}
                className={`h-auto py-3 sm:py-4 flex-col gap-1 sm:gap-2 border-2 text-xs sm:text-sm ${
                  currentClassification?.type === "unsure"
                    ? "bg-chart-5 text-white border-chart-5 hover:bg-chart-5/90"
                    : "border-border bg-background text-foreground hover:bg-muted"
                }`}
                onClick={() => handleClassify("unsure")}
              >
                <HelpCircle className="w-5 h-5 sm:w-6 sm:h-6" />
                <span className="font-semibold">Unsure</span>
              </Button>
            </div>

            {currentClassification?.comment && (
              <div className="bg-accent/10 border-2 border-accent/30 rounded-lg p-3">
                <p className="text-xs font-medium text-muted-foreground mb-1">Saved Comment:</p>
                <p className="text-sm text-foreground">{currentClassification.comment}</p>
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </Card>
  )
}
