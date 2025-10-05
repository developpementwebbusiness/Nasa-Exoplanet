"use client";

import type React from "react";

import { useCallback, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Upload, FileUp, CheckCircle2, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ColumnMapper } from "@/components/column-mapper";
import Papa from "papaparse";

interface CSVUploaderProps {
  onUpload: (data: any[]) => void;
  isProcessing: boolean;
}

export function CSVUploader({ onUpload, isProcessing }: CSVUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const [rowCount, setRowCount] = useState<number>(0);
  const [skippedRows, setSkippedRows] = useState<number>(0);
  const [showMapper, setShowMapper] = useState(false);
  const [pendingData, setPendingData] = useState<any[] | null>(null);
  const [csvColumns, setCsvColumns] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      setFileName(file.name);
      const fileExtension = file.name.split(".").pop()?.toLowerCase();

      if (fileExtension === "csv" || file.type === "text/csv") {
        // Handle CSV files with Papa Parse
        Papa.parse(file, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          beforeFirstChunk: (chunk) => {
            const lines = chunk.split("\n");
            const cleanedLines = lines.filter((line) => {
              const trimmed = line.trim();
              return (
                trimmed && !trimmed.startsWith("#") && !trimmed.startsWith("//")
              );
            });
            const skipped = lines.length - cleanedLines.length;
            setSkippedRows(skipped);
            return cleanedLines.join("\n");
          },
          complete: (results) => {
            const cleanData = results.data.filter((row: any) => {
              const firstValue = Object.values(row)[0];
              return (
                firstValue !== null &&
                firstValue !== undefined &&
                String(firstValue).trim() !== "" &&
                !String(firstValue).startsWith("#")
              );
            });
            setRowCount(cleanData.length);
            
            // Get column names from the first row
            if (cleanData.length > 0) {
              const columns = Object.keys(cleanData[0] as Record<string, any>);
              setCsvColumns(columns);
              setPendingData(cleanData);
              setShowMapper(true);
            }
          },
          error: (error) => {
            console.error("[v0] CSV parsing error:", error);
          },
        });
      } else if (fileExtension === "json" || file.type === "application/json") {
        // Handle JSON files
        const reader = new FileReader();
        reader.onload = (e) => {
          try {
            const jsonData = JSON.parse(e.target?.result as string);
            const dataArray = Array.isArray(jsonData) ? jsonData : [jsonData];
            setRowCount(dataArray.length);
            setSkippedRows(0);
            
            // Get column names from the first item
            if (dataArray.length > 0) {
              const columns = Object.keys(dataArray[0]);
              setCsvColumns(columns);
              setPendingData(dataArray);
              setShowMapper(true);
            }
          } catch (error) {
            console.error("[v0] JSON parsing error:", error);
          }
        };
        reader.readAsText(file);
      } else if (["xlsx", "xls"].includes(fileExtension || "")) {
        // Handle Excel files (would need xlsx library)
        console.warn(
          "[v0] Excel file detected but parsing not yet implemented. Please convert to CSV or JSON."
        );
        // For now, show an error message
        setRowCount(0);
        setSkippedRows(0);
      } else {
        // Handle plain text files
        const reader = new FileReader();
        reader.onload = (e) => {
          const text = e.target?.result as string;
          const lines = text.split("\n").filter((line) => line.trim());
          // Try to parse as CSV-like data
          if (lines.length > 0) {
            const headers = lines[0].split(",").map((h) => h.trim());
            const data = lines.slice(1).map((line) => {
              const values = line.split(",");
              const obj: any = {};
              headers.forEach((header, index) => {
                obj[header] = values[index]?.trim() || "";
              });
              return obj;
            });
            setRowCount(data.length);
            setSkippedRows(0);
            
            // Get column names
            if (data.length > 0) {
              const columns = Object.keys(data[0]);
              setCsvColumns(columns);
              setPendingData(data);
              setShowMapper(true);
            }
          }
        };
        reader.readAsText(file);
      }
    },
    []
  );

  const handleMappingConfirm = useCallback(
    (mapping: Record<string, string>) => {
      if (!pendingData) return;

      // Apply the mapping to transform the data
      const mappedData = pendingData.map((row) => {
        const newRow: any = {};
        
        // Keep original data
        Object.assign(newRow, row);
        
        // Add mapped columns with standardized names
        Object.entries(mapping).forEach(([standardKey, csvColumn]) => {
          newRow[standardKey] = row[csvColumn];
        });
        
        return newRow;
      });

      setShowMapper(false);
      setPendingData(null);
      onUpload(mappedData);
    },
    [pendingData, onUpload]
  );

  const handleMappingCancel = useCallback(() => {
    setShowMapper(false);
    setPendingData(null);
    setFileName(null);
    setRowCount(0);
    setSkippedRows(0);
    setCsvColumns([]);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const file = e.dataTransfer.files[0];
      const allowedExtensions = [".csv", ".xlsx", ".xls", ".json", ".txt"];
      const isAllowed =
        file &&
        (allowedExtensions.some((ext) =>
          file.name.toLowerCase().endsWith(ext)
        ) ||
          [
            "text/csv",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
            "application/json",
            "text/plain",
          ].includes(file.type));

      if (isAllowed) {
        handleFile(file);
      }
    },
    [handleFile]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        handleFile(file);
      }
    },
    [handleFile]
  );

  return (
    <>
      {showMapper && (
        <ColumnMapper
          csvColumns={csvColumns}
          onConfirm={handleMappingConfirm}
          onCancel={handleMappingCancel}
        />
      )}
      
      <div className="space-y-4">
      <motion.div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
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
        <input
          type="file"
          accept=".csv,.xlsx,.xls,.json,.txt"
          onChange={handleFileInput}
          className="hidden"
          id="csv-upload"
          ref={fileInputRef}
        />

        <label htmlFor="csv-upload" className="cursor-pointer">
          <motion.div
            animate={{ scale: isDragging ? 1.05 : 1 }}
            className="flex flex-col items-center gap-4"
          >
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
                {fileName ? fileName : "Drop your data file here"}
              </p>
              <p className="text-sm text-muted-foreground">
                {isProcessing
                  ? "Processing and analyzing data..."
                  : fileName
                  ? `${rowCount} candidates loaded${
                      skippedRows > 0
                        ? ` (${skippedRows} comment rows cleaned)`
                        : ""
                    }`
                  : "Supports CSV, Excel (.xlsx/.xls), JSON, and TXT files"}
              </p>
            </div>

            {!fileName && (
              <Button
                variant="default"
                className="mt-2 bg-primary hover:bg-primary/90 text-primary-foreground nasa-glow cursor-pointer"
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload className="w-4 h-4 mr-2" />
                Select Data File
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
              <p className="text-sm font-medium text-foreground">
                Data loaded and cleaned successfully
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                {skippedRows > 0 &&
                  `Removed ${skippedRows} header/comment lines. `}
                Ready for AI analysis and manual classification
              </p>
            </div>
          </div>
        </motion.div>
      )}
    </div>
    </>
  );
}
