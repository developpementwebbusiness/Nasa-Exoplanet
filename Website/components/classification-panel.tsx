"use client";

import { motion } from "framer-motion";
import {
  CheckCircle2,
  XCircle,
  HelpCircle,
  ChevronLeft,
  ChevronRight,
  MessageSquare,
  Hash,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { useState, useEffect } from "react";

interface ClassificationWithComment {
  type: "exoplanet" | "not_exoplanet" | "unsure";
  comment?: string;
}

interface ClassificationPanelProps {
  csvData: any[];
  predictions: any[];
  classifications: Record<number, ClassificationWithComment>;
  onClassify: (
    index: number,
    classification: "exoplanet" | "not_exoplanet" | "unsure",
    comment?: string
  ) => void;
  selectedCandidate: number | null;
  onSelectCandidate: (index: number) => void;
}

export function ClassificationPanel({
  csvData,
  predictions,
  classifications,
  onClassify,
  selectedCandidate,
  onSelectCandidate,
}: ClassificationPanelProps) {
  const currentIndex = selectedCandidate ?? 0;
  const currentData = csvData[currentIndex];
  const currentPrediction = predictions[currentIndex];
  const currentClassification = classifications[currentIndex];

  const [comment, setComment] = useState(currentClassification?.comment || "");
  const [customIndex, setCustomIndex] = useState<string>("");
  const [debounceTimeout, setDebounceTimeout] = useState<NodeJS.Timeout | null>(
    null
  );
  const [localClassification, setLocalClassification] = useState<
    "exoplanet" | "not_exoplanet" | "unsure" | null
  >(currentClassification?.type || null);

  // Update comment when candidate changes
  useEffect(() => {
    setComment(currentClassification?.comment || "");
    setLocalClassification(currentClassification?.type || null);
  }, [currentIndex, currentClassification]);

  const handlePrevious = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (currentIndex > 0) {
      onSelectCandidate(currentIndex - 1);
    }
  };

  const handleNext = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (currentIndex < csvData.length - 1) {
      onSelectCandidate(currentIndex + 1);
    }
  };

  const handleCustomIndexChange = (value: string) => {
    // Only allow numbers
    const filtered = value.replace(/[^0-9]/g, "");
    setCustomIndex(filtered);

    // Clear previous timeout
    if (debounceTimeout) {
      clearTimeout(debounceTimeout);
    }

    // Set new timeout - wait 500ms after user stops typing
    const timeout = setTimeout(() => {
      const num = parseInt(filtered);
      if (!isNaN(num) && num >= 1 && num <= csvData.length) {
        onSelectCandidate(num - 1);
      }
    }, 500);

    setDebounceTimeout(timeout);
  };

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (debounceTimeout) {
        clearTimeout(debounceTimeout);
      }
    };
  }, [debounceTimeout]);

  const handleClassify = (type: "exoplanet" | "not_exoplanet" | "unsure") => {
    // Update local state immediately for instant visual feedback
    setLocalClassification(type);
    // Update parent state
    onClassify(currentIndex, type, comment);
  };

  if (csvData.length === 0) {
    return (
      <Card className="p-8 text-center bg-card border-2 border-border">
        <p className="text-muted-foreground">
          Upload CSV data to start classification
        </p>
      </Card>
    );
  }

  const confidence = currentPrediction
    ? (
        (currentPrediction.score || currentPrediction.confidence || currentPrediction.probability || 0) *
        100
      ).toFixed(2)
    : "N/A";

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

          <div className="flex-1 text-center max-w-xs">
            <p className="text-xs sm:text-sm text-muted-foreground mb-1">
              Candidate
            </p>
            <div className="flex items-center justify-center gap-2">
              <div className="relative">
                <Hash className="absolute left-2 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
                <Input
                  type="text"
                  inputMode="numeric"
                  pattern="[0-9]*"
                  value={customIndex || currentIndex + 1}
                  onChange={(e) => handleCustomIndexChange(e.target.value)}
                  onFocus={() => setCustomIndex(String(currentIndex + 1))}
                  onBlur={() => setCustomIndex("")}
                  placeholder={String(currentIndex + 1)}
                  className="w-20 h-9 pl-7 pr-2 text-center font-bold text-primary border-2 border-primary/30 focus:border-primary"
                />
              </div>
              <span className="text-sm text-muted-foreground font-medium">
                / {csvData.length}
              </span>
            </div>
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
            <p className="text-sm font-semibold text-foreground">
              AI Prediction Confidence
            </p>
            <Badge
              className={
                Number.parseFloat(confidence) > 50
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-muted-foreground"
              }
            >
              {confidence}%
            </Badge>
          </div>
          <div className="w-full bg-muted rounded-full h-3 border border-border">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${confidence}%` }}
              transition={{ duration: 0.8, ease: "easeOut" }}
              className="bg-primary h-3 rounded-full"
            />
          </div>
        </div>

        {/* Data Preview */}
        <div className="bg-card border-2 border-border rounded-lg p-4">
          <p className="text-sm font-semibold text-foreground mb-3">
            Candidate Data Preview
          </p>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3 max-h-64 overflow-y-auto">
            {currentData &&
              Object.entries(currentData)
                .slice(0, 15)
                .map(([key, value]: [string, any]) => (
                  <div
                    key={key}
                    className="text-xs bg-muted/50 p-2 rounded border border-border"
                  >
                    <p className="text-muted-foreground truncate font-medium">
                      {key}
                    </p>
                    <p className="font-mono text-foreground font-semibold">
                      {typeof value === "number"
                        ? value.toFixed(4)
                        : String(value)}
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
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-3">
            <Button
              className={`h-14 sm:h-16 flex-col gap-1.5 border-2 transition-none no-hover-scale ${
                localClassification === "exoplanet"
                  ? "bg-green-500 text-white border-green-600 shadow-lg hover:bg-green-600 hover:border-green-700"
                  : "bg-green-500/10 text-green-400 border-green-600/30 hover:bg-green-500/25 hover:border-green-600/60"
              }`}
              onClick={() => handleClassify("exoplanet")}
            >
              <CheckCircle2 className="w-5 h-5" />
              <span className="font-semibold text-xs">Exoplanet</span>
            </Button>

            <Button
              className={`h-14 sm:h-16 flex-col gap-1.5 border-2 transition-none no-hover-scale ${
                localClassification === "not_exoplanet"
                  ? "bg-red-500 text-white border-red-600 shadow-lg hover:bg-red-600 hover:border-red-700"
                  : "bg-red-500/10 text-red-400 border-red-600/30 hover:bg-red-500/25 hover:border-red-600/60"
              }`}
              onClick={() => handleClassify("not_exoplanet")}
            >
              <XCircle className="w-5 h-5" />
              <span className="font-semibold text-xs">Not Exoplanet</span>
            </Button>

            <Button
              className={`h-14 sm:h-16 flex-col gap-1.5 border-2 transition-none no-hover-scale ${
                localClassification === "unsure"
                  ? "bg-yellow-500 text-white border-yellow-600 shadow-lg hover:bg-yellow-600 hover:border-yellow-700"
                  : "bg-yellow-500/10 text-yellow-400 border-yellow-600/30 hover:bg-yellow-500/25 hover:border-yellow-600/60"
              }`}
              onClick={() => handleClassify("unsure")}
            >
              <HelpCircle className="w-5 h-5" />
              <span className="font-semibold text-xs">Unsure</span>
            </Button>
          </div>

          {currentClassification?.comment && (
            <div className="bg-accent/10 border-2 border-accent/30 rounded-lg p-3">
              <p className="text-xs font-medium text-muted-foreground mb-1">
                Saved Comment:
              </p>
              <p className="text-sm text-foreground">
                {currentClassification.comment}
              </p>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}
