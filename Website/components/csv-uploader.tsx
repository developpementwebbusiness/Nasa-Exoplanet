"use client"

import type React from "react"

import { useCallback, useState } from "react"
import { motion } from "framer-motion"
import { Upload, FileUp, CheckCircle2, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import Papa from "papaparse"

interface CSVUploaderProps {
  onUpload: (data: any[]) => void
  isProcessing: boolean
}

export function CSVUploader({ onUpload, isProcessing }: CSVUploaderProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [fileName, setFileName] = useState<string | null>(null)
  const [rowCount, setRowCount] = useState<number>(0)
  const [skippedRows, setSkippedRows] = useState<number>(0)

  const handleFile = useCallback(
    (file: File) => {
      setFileName(file.name)

      Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        beforeFirstChunk: (chunk) => {
          const lines = chunk.split("\n")
          const cleanedLines = lines.filter((line) => {
            const trimmed = line.trim()
            return trimmed && !trimmed.startsWith("#") && !trimmed.startsWith("//")
          })
          const skipped = lines.length - cleanedLines.length
          setSkippedRows(skipped)
          return cleanedLines.join("\n")
        },
        complete: (results) => {
          const cleanData = results.data.filter((row: any) => {
            const firstValue = Object.values(row)[0]
            return (
              firstValue !== null &&
              firstValue !== undefined &&
              String(firstValue).trim() !== "" &&
              !String(firstValue).startsWith("#")
            )
          })
          setRowCount(cleanData.length)
          onUpload(cleanData)
        },
        error: (error) => {
          console.error("[v0] CSV parsing error:", error)
        },
      })
    },
    [onUpload],
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)

      const file = e.dataTransfer.files[0]
      if (file && (file.type === "text/csv" || file.name.endsWith(".csv"))) {
        handleFile(file)
      }
    },
    [handleFile],
  )

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) {
        handleFile(file)
      }
    },
    [handleFile],
  )

  return (
    <div className="space-y-4">
      <motion.div
        onDragOver={(e) => {
          e.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        animate={{
          scale: isDragging ? 1.02 : 1,
        }}
        className={`border-2 border-dashed rounded-lg p-12 text-center transition-all bg-card shadow-lg ${
          isDragging ? "border-primary bg-primary/5" : "border-border"
        }`}
      >
        <input type="file" accept=".csv" onChange={handleFileInput} className="hidden" id="csv-upload" />

        <label htmlFor="csv-upload" className="cursor-pointer">
          <motion.div animate={{ scale: isDragging ? 1.05 : 1 }} className="flex flex-col items-center gap-4">
            <div className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center border-2 border-primary/30 nasa-glow">
              {isProcessing ? (
                <Loader2 className="w-10 h-10 text-primary animate-spin" />
              ) : fileName ? (
                <CheckCircle2 className="w-10 h-10 text-chart-4" />
              ) : (
                <FileUp className="w-10 h-10 text-primary" />
              )}
            </div>

            <div>
              <p className="text-lg font-semibold text-foreground mb-1">
                {fileName ? fileName : "Drop your Kepler CSV file here"}
              </p>
              <p className="text-sm text-muted-foreground">
                {isProcessing
                  ? "Processing and analyzing data..."
                  : fileName
                    ? `${rowCount} candidates loaded${skippedRows > 0 ? ` (${skippedRows} comment rows cleaned)` : ""}`
                    : "or click to browse your files"}
              </p>
            </div>

            {!fileName && (
              <Button
                variant="default"
                className="mt-2 bg-primary hover:bg-primary/90 text-primary-foreground nasa-glow"
              >
                <Upload className="w-4 h-4 mr-2" />
                Select CSV File
              </Button>
            )}
          </motion.div>
        </label>
      </motion.div>

      {fileName && !isProcessing && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-chart-4/10 border-2 border-chart-4/30 rounded-lg p-4"
        >
          <div className="flex items-start gap-3">
            <CheckCircle2 className="w-5 h-5 text-chart-4 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm font-medium text-foreground">Data loaded and cleaned successfully</p>
              <p className="text-xs text-muted-foreground mt-1">
                {skippedRows > 0 && `Removed ${skippedRows} header/comment lines. `}
                Ready for AI analysis and manual classification
              </p>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}
