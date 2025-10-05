"use client";

import { useState, useMemo, useCallback } from "react";
import {
  Search,
  ChevronDown,
  ChevronUp,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface ClassificationWithComment {
  type: "exoplanet" | "not_exoplanet" | "unsure";
  comment?: string;
}

interface DataTableProps {
  data: any[];
  predictions: any[];
  classifications: Record<number, ClassificationWithComment>;
  onSelectCandidate: (index: number) => void;
  selectedCandidate: number | null;
}

const ITEMS_PER_PAGE = 50;
const MISSING_DATA_THRESHOLD = 0.7; // Hide columns with more than 70% missing data

export function DataTable({
  data,
  predictions,
  classifications,
  onSelectCandidate,
  selectedCandidate,
}: DataTableProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [filterType, setFilterType] = useState<string>("all");
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(ITEMS_PER_PAGE);
  const [customInput, setCustomInput] = useState(String(ITEMS_PER_PAGE));
  const [debounceTimeout, setDebounceTimeout] = useState<NodeJS.Timeout | null>(
    null
  );

  const columns = useMemo(() => {
    if (data.length === 0) return [];

    const allKeys = Object.keys(data[0]);

    // Calculate missing data percentage for each column
    const columnStats = allKeys.map((key) => {
      const missingCount = data.filter((row) => {
        const value = row[key];
        return value === null || value === undefined || value === "";
      }).length;

      const missingPercentage = missingCount / data.length;

      return {
        key,
        missingPercentage,
        hasData: missingCount < data.length,
      };
    });

    // Only keep columns that have data and are below the missing threshold
    return columnStats
      .filter(
        (stat) =>
          stat.hasData && stat.missingPercentage < MISSING_DATA_THRESHOLD
      )
      .map((stat) => stat.key);
  }, [data]);

  const filteredData = useMemo(() => {
    let filtered = data.map((row, index) => ({
      ...row,
      _index: index,
      _confidence:
        predictions[index]?.score || predictions[index]?.confidence || predictions[index]?.probability || 0,
      _classification: classifications[index]?.type || "unclassified",
    }));

    if (searchTerm) {
      filtered = filtered.filter((row) =>
        Object.values(row).some((val) =>
          String(val).toLowerCase().includes(searchTerm.toLowerCase())
        )
      );
    }

    if (filterType !== "all") {
      filtered = filtered.filter((row) => row._classification === filterType);
    }

    if (sortColumn) {
      filtered.sort((a, b) => {
        const aVal = a[sortColumn];
        const bVal = b[sortColumn];
        if (typeof aVal === "number" && typeof bVal === "number") {
          return sortDirection === "asc" ? aVal - bVal : bVal - aVal;
        }
        return sortDirection === "asc"
          ? String(aVal).localeCompare(String(bVal))
          : String(bVal).localeCompare(String(aVal));
      });
    }

    return filtered;
  }, [
    data,
    predictions,
    classifications,
    searchTerm,
    filterType,
    sortColumn,
    sortDirection,
  ]);

  const totalPages = Math.ceil(filteredData.length / itemsPerPage);
  const paginatedData = useMemo(() => {
    const start = (currentPage - 1) * itemsPerPage;
    return filteredData.slice(start, start + itemsPerPage);
  }, [filteredData, currentPage, itemsPerPage]);

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortColumn(column);
      setSortDirection("asc");
    }
    setCurrentPage(1);
  };

  const handleCustomInputChange = useCallback(
    (value: string) => {
      setCustomInput(value);

      // Clear existing timeout
      if (debounceTimeout) {
        clearTimeout(debounceTimeout);
      }

      // Set new timeout - wait 500ms after user stops typing
      const timeout = setTimeout(() => {
        const numValue = parseInt(value);
        if (value && numValue > 0 && numValue <= filteredData.length) {
          setItemsPerPage(numValue);
          setCurrentPage(1);
        } else if (!value) {
          setItemsPerPage(ITEMS_PER_PAGE);
          setCustomInput(String(ITEMS_PER_PAGE));
          setCurrentPage(1);
        }
      }, 500);

      setDebounceTimeout(timeout);
    },
    [debounceTimeout, filteredData.length]
  );

  const handlePresetClick = (size: number) => {
    setItemsPerPage(size);
    setCustomInput(String(size));
    setCurrentPage(1);
  };

  const getClassificationBadge = (classification: string) => {
    switch (classification) {
      case "exoplanet":
        return (
          <Badge className="bg-chart-4 text-white border-0">Exoplanet</Badge>
        );
      case "not_exoplanet":
        return (
          <Badge className="bg-chart-2 text-white border-0">
            Not Exoplanet
          </Badge>
        );
      case "unsure":
        return <Badge className="bg-chart-5 text-white border-0">Unsure</Badge>;
      default:
        return (
          <Badge
            variant="outline"
            className="border-muted-foreground/30 text-muted-foreground"
          >
            Unclassified
          </Badge>
        );
    }
  };

  return (
    <Card className="p-6 bg-card border-2 border-primary/30 shadow-lg">
      <div className="flex flex-col md:flex-row gap-4 mb-6">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search candidates..."
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setCurrentPage(1);
            }}
            className="pl-10 border-2 border-primary/20 focus:border-primary bg-background text-foreground"
          />
        </div>
        <Select
          value={filterType}
          onValueChange={(v) => {
            setFilterType(v);
            setCurrentPage(1);
          }}
        >
          <SelectTrigger className="w-full md:w-[200px] border-2 border-primary/20 bg-background text-foreground">
            <SelectValue placeholder="Filter by classification" />
          </SelectTrigger>
          <SelectContent className="bg-popover text-popover-foreground border-border">
            <SelectItem value="all">All Candidates</SelectItem>
            <SelectItem value="exoplanet">Exoplanets</SelectItem>
            <SelectItem value="not_exoplanet">Not Exoplanets</SelectItem>
            <SelectItem value="unsure">Unsure</SelectItem>
            <SelectItem value="unclassified">Unclassified</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="overflow-x-auto rounded-lg border-2 border-primary/20">
        <div className="h-[500px] overflow-y-auto">
          {/* Fixed height for consistent size */}
          <table className="w-full">
            <thead className="sticky top-0 bg-primary text-primary-foreground z-10 shadow-md">
              <tr>
                <th className="text-left p-3 text-sm font-semibold">#</th>
                {columns.map((col) => (
                  <th
                    key={col}
                    className="text-left p-3 text-sm font-semibold cursor-pointer hover:bg-primary/80 transition-colors"
                    onClick={() => handleSort(col)}
                  >
                    <div className="flex items-center gap-2">
                      {col}
                      {sortColumn === col &&
                        (sortDirection === "asc" ? (
                          <ChevronUp className="w-4 h-4" />
                        ) : (
                          <ChevronDown className="w-4 h-4" />
                        ))}
                    </div>
                  </th>
                ))}
                <th className="text-left p-3 text-sm font-semibold">
                  Confidence
                </th>
                <th className="text-left p-3 text-sm font-semibold">
                  Classification
                </th>
              </tr>
            </thead>
            <tbody className="bg-card">
              {paginatedData.length === 0 ? (
                <tr>
                  <td
                    colSpan={columns.length + 3}
                    className="text-center p-8 text-muted-foreground"
                  >
                    No candidates found
                  </td>
                </tr>
              ) : (
                <>
                  {paginatedData.map((row) => (
                    <tr
                      key={row._index}
                      onClick={() => onSelectCandidate(row._index)}
                      className={`border-b border-border hover:bg-primary/10 transition-colors cursor-pointer ${
                        selectedCandidate === row._index
                          ? "bg-primary/10 border-primary/30"
                          : ""
                      }`}
                    >
                      <td className="p-3 text-sm text-muted-foreground font-semibold">
                        {row._index + 1}
                      </td>
                      {columns.map((col) => {
                        const value = row[col];
                        // Skip rendering null/undefined/empty values
                        if (
                          value === null ||
                          value === undefined ||
                          value === ""
                        ) {
                          return (
                            <td
                              key={col}
                              className="p-3 text-sm text-muted-foreground"
                            >
                              N/A
                            </td>
                          );
                        }

                        return (
                          <td
                            key={col}
                            className="p-3 text-sm text-foreground font-mono max-w-xs truncate"
                            title={
                              typeof value === "number"
                                ? value.toFixed(6)
                                : String(value)
                            }
                          >
                            {typeof value === "number"
                              ? value.toFixed(4)
                              : String(value).substring(0, 50) +
                                (String(value).length > 50 ? "..." : "")}
                          </td>
                        );
                      })}
                      <td className="p-3 text-sm">
                        <Badge
                          variant="outline"
                          className="font-mono border-primary/30 text-foreground bg-muted"
                        >
                          {(row._confidence * 100).toFixed(1)}%
                        </Badge>
                      </td>
                      <td className="p-3 text-sm">
                        {getClassificationBadge(row._classification)}
                      </td>
                    </tr>
                  ))}
                </>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="mt-4 flex flex-col sm:flex-row items-center justify-between gap-4">
        {/* Items per page selector */}
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          <span className="text-sm text-muted-foreground">Show:</span>
          <div className="flex gap-2">
            {[50, 100, 200, 500].map((size) => (
              <Button
                key={size}
                variant={itemsPerPage === size ? "default" : "outline"}
                size="sm"
                onClick={() => handlePresetClick(size)}
                className={
                  itemsPerPage === size
                    ? "bg-primary text-primary-foreground"
                    : "border-primary/20"
                }
              >
                {size}
              </Button>
            ))}
            <Input
              type="text"
              placeholder="Custom"
              value={customInput}
              onChange={(e) => {
                const value = e.target.value.replace(/[^0-9]/g, ""); // Only allow numbers
                handleCustomInputChange(value);
              }}
              onBlur={(e) => {
                // Reset to current itemsPerPage if invalid on blur
                if (!e.target.value || parseInt(e.target.value) < 1) {
                  setCustomInput(String(itemsPerPage));
                }
              }}
              className="w-24 h-8 text-center border-2 border-primary/20 focus:border-primary"
            />
          </div>
          <span className="text-sm text-muted-foreground">per page</span>
        </div>

        {/* Pagination controls */}
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setCurrentPage(1)}
            disabled={currentPage === 1}
            className="border-primary/30"
          >
            First
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            disabled={currentPage === 1}
            className="border-primary/30"
          >
            <ChevronLeft className="w-4 h-4" />
          </Button>
          <span className="text-sm text-foreground font-medium px-2">
            Page {currentPage} of {totalPages}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
            disabled={currentPage === totalPages}
            className="border-primary/30"
          >
            <ChevronRight className="w-4 h-4" />
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setCurrentPage(totalPages)}
            disabled={currentPage === totalPages}
            className="border-primary/30"
          >
            Last
          </Button>
        </div>
      </div>
    </Card>
  );
}
