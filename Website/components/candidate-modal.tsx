"use client";

import { X, CheckCircle2, XCircle, HelpCircle, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { useState, useEffect } from "react";

interface ClassificationWithComment {
  type: "exoplanet" | "not_exoplanet" | "unsure";
  comment?: string;
}

interface CandidateModalProps {
  isOpen: boolean;
  onClose: () => void;
  candidateData: any;
  candidateIndex: number;
  prediction: any;
  classification?: ClassificationWithComment;
  onClassify: (
    index: number,
    classification: "exoplanet" | "not_exoplanet" | "unsure",
    comment?: string
  ) => void;
}

export function CandidateModal({
  isOpen,
  onClose,
  candidateData,
  candidateIndex,
  prediction,
  classification,
  onClassify,
}: CandidateModalProps) {
  const [comment, setComment] = useState(classification?.comment || "");
  const [localClassification, setLocalClassification] = useState<
    "exoplanet" | "not_exoplanet" | "unsure" | null
  >(classification?.type || null);

  // Handle Escape key press
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener("keydown", handleEscape);
      // Prevent body scroll when modal is open
      document.body.style.overflow = "hidden";
    }

    return () => {
      document.removeEventListener("keydown", handleEscape);
      document.body.style.overflow = "unset";
    };
  }, [isOpen, onClose]);

  // Update local state when classification changes
  useEffect(() => {
    setLocalClassification(classification?.type || null);
    setComment(classification?.comment || "");
  }, [classification]);

  if (!isOpen || !candidateData) return null;

  const handleClassify = (type: "exoplanet" | "not_exoplanet" | "unsure") => {
    // Update local state immediately for instant feedback
    setLocalClassification(type);
    onClassify(candidateIndex, type, comment);
  };

  const renderValue = (value: any) => {
    if (value === null || value === undefined) return "N/A";
    if (typeof value === "string" && value.includes("href=")) {
      // Handle HTML content - render as text to prevent overflow
      return (
        <div
          className="text-sm break-all max-w-full overflow-hidden"
          dangerouslySetInnerHTML={{ __html: value }}
        />
      );
    }
    if (typeof value === "number") return value.toFixed(6);
    return String(value);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-2 sm:p-4 bg-black/60 backdrop-blur-md"
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-6xl h-[96vh] flex flex-col animate-in fade-in zoom-in-95 duration-300"
      >
        <Card className="bg-card border border-border/50 shadow-2xl rounded-2xl flex flex-col h-full overflow-hidden">
          {/* Header - Fixed */}
          <div className="bg-card border-b border-border/50 px-4 sm:px-6 py-3 sm:py-4 flex-shrink-0">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 sm:gap-4">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                  <h2 className="text-lg sm:text-2xl font-bold text-foreground">
                    Candidate #{candidateIndex + 1}
                  </h2>
                </div>
                <Badge
                  variant="outline"
                  className="bg-primary/10 text-primary border-primary/30 px-2 sm:px-3 py-0.5 sm:py-1 text-xs sm:text-sm font-semibold"
                >
                  {(
                    (prediction?.score || prediction?.confidence || prediction?.probability || 0) *
                    100
                  ).toFixed(1)}
                  % AI
                </Badge>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={onClose}
                className="hover:bg-destructive/10 hover:text-destructive rounded-full transition-all h-8 w-8 sm:h-10 sm:w-10"
              >
                <X className="w-4 h-4 sm:w-5 sm:h-5" />
              </Button>
            </div>
          </div>

          {/* Content - Scrollable */}
          <div className="p-3 sm:p-6 overflow-y-auto flex-1">
            <div className="space-y-4 sm:space-y-6">
              {/* Data Grid */}
              <div>
                <h3 className="text-xs sm:text-sm font-semibold text-muted-foreground uppercase tracking-wide mb-3 sm:mb-4">
                  Stellar Data
                </h3>
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-2 sm:gap-3">
                  {Object.entries(candidateData).map(([key, value]) => {
                    if (key.startsWith("_")) return null;
                    if (value === null || value === undefined || value === "")
                      return null;

                    return (
                      <div
                        key={key}
                        className="bg-muted/40 rounded-lg sm:rounded-xl p-2 sm:p-3 border border-border/40 hover:border-primary/30 transition-colors"
                      >
                        <p className="text-[10px] sm:text-xs font-medium text-muted-foreground mb-1 sm:mb-1.5 uppercase tracking-wide truncate">
                          {key}
                        </p>
                        <div className="text-xs sm:text-sm text-foreground font-semibold break-words font-mono">
                          {renderValue(value)}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Classification Section */}
              <div className="border-t border-border/50 pt-4 sm:pt-6">
                <h3 className="text-xs sm:text-sm font-semibold text-muted-foreground uppercase tracking-wide mb-3 sm:mb-4">
                  Your Classification
                </h3>
                <div className="grid grid-cols-3 gap-2 sm:gap-3 mb-3 sm:mb-4">
                  <Button
                    onClick={() => handleClassify("exoplanet")}
                    className={`h-14 sm:h-16 flex-col gap-1.5 border-2 transition-none no-hover-scale ${
                      localClassification === "exoplanet"
                        ? "bg-green-500 text-white border-green-600 shadow-lg hover:bg-green-600 hover:border-green-700"
                        : "bg-green-500/10 text-green-400 border-green-600/30 hover:bg-green-500/25 hover:border-green-600/60"
                    }`}
                  >
                    <CheckCircle2 className="w-5 h-5" />
                    <span className="font-semibold text-xs">Exoplanet</span>
                  </Button>
                  <Button
                    onClick={() => handleClassify("not_exoplanet")}
                    className={`h-14 sm:h-16 flex-col gap-1.5 border-2 transition-none no-hover-scale ${
                      localClassification === "not_exoplanet"
                        ? "bg-red-500 text-white border-red-600 shadow-lg hover:bg-red-600 hover:border-red-700"
                        : "bg-red-500/10 text-red-400 border-red-600/30 hover:bg-red-500/25 hover:border-red-600/60"
                    }`}
                  >
                    <XCircle className="w-5 h-5" />
                    <span className="font-semibold text-xs">Not Exoplanet</span>
                  </Button>
                  <Button
                    onClick={() => handleClassify("unsure")}
                    className={`h-14 sm:h-16 flex-col gap-1.5 border-2 transition-none no-hover-scale ${
                      localClassification === "unsure"
                        ? "bg-yellow-500 text-white border-yellow-600 shadow-lg hover:bg-yellow-600 hover:border-yellow-700"
                        : "bg-yellow-500/10 text-yellow-400 border-yellow-600/30 hover:bg-yellow-500/25 hover:border-yellow-600/60"
                    }`}
                  >
                    <HelpCircle className="w-5 h-5" />
                    <span className="font-semibold text-xs">Unsure</span>
                  </Button>
                </div>

                <div>
                  <label className="text-xs sm:text-sm font-medium text-foreground mb-2 block">
                    Notes
                  </label>
                  <Textarea
                    value={comment}
                    onChange={(e) => setComment(e.target.value)}
                    placeholder="Add your observations and notes..."
                    className="min-h-[80px] sm:min-h-[100px] resize-none border-2 border-border focus:border-primary rounded-lg sm:rounded-xl bg-background text-sm"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Footer - Fixed */}
          <div className="border-t border-border/50 px-4 sm:px-6 py-3 sm:py-4 bg-muted/20 flex-shrink-0">
            <div className="flex justify-between items-center gap-2">
              <p className="text-xs sm:text-sm text-muted-foreground hidden sm:block">
                Press{" "}
                <kbd className="px-2 py-1 text-xs bg-muted border border-border rounded">
                  Esc
                </kbd>{" "}
                to close
              </p>
              <Button
                variant="outline"
                onClick={onClose}
                className="px-4 sm:px-6 hover:bg-muted text-sm ml-auto"
              >
                Close
              </Button>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
