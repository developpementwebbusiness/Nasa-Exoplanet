"use client"

import { useState, useMemo } from "react"
import { motion } from "framer-motion"
import { Search, ChevronDown, ChevronUp, Eye, ChevronLeft, ChevronRight } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

interface ClassificationWithComment {
  type: "exoplanet" | "not_exoplanet" | "unsure"
  comment?: string
}

interface DataTableProps {
  data: any[]
  predictions: any[]
  classifications: Record<number, ClassificationWithComment>
  onSelectCandidate: (index: number) => void
}

const ITEMS_PER_PAGE = 50

export function DataTable({ data, predictions, classifications, onSelectCandidate }: DataTableProps) {
  const [searchTerm, setSearchTerm] = useState("")
  const [filterType, setFilterType] = useState<string>("all")
  const [sortColumn, setSortColumn] = useState<string | null>(null)
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc")
  const [currentPage, setCurrentPage] = useState(1)

  const columns = useMemo(() => {
    if (data.length === 0) return []
    return Object.keys(data[0]).slice(0, 8)
  }, [data])

  const filteredData = useMemo(() => {
    let filtered = data.map((row, index) => ({
      ...row,
      _index: index,
      _probability: predictions[index]?.probability || 0,
      _classification: classifications[index]?.type || "unclassified",
    }))

    if (searchTerm) {
      filtered = filtered.filter((row) =>
        Object.values(row).some((val) => String(val).toLowerCase().includes(searchTerm.toLowerCase())),
      )
    }

    if (filterType !== "all") {
      filtered = filtered.filter((row) => row._classification === filterType)
    }

    if (sortColumn) {
      filtered.sort((a, b) => {
        const aVal = a[sortColumn]
        const bVal = b[sortColumn]
        if (typeof aVal === "number" && typeof bVal === "number") {
          return sortDirection === "asc" ? aVal - bVal : bVal - aVal
        }
        return sortDirection === "asc"
          ? String(aVal).localeCompare(String(bVal))
          : String(bVal).localeCompare(String(aVal))
      })
    }

    return filtered
  }, [data, predictions, classifications, searchTerm, filterType, sortColumn, sortDirection])

  const totalPages = Math.ceil(filteredData.length / ITEMS_PER_PAGE)
  const paginatedData = useMemo(() => {
    const start = (currentPage - 1) * ITEMS_PER_PAGE
    return filteredData.slice(start, start + ITEMS_PER_PAGE)
  }, [filteredData, currentPage])

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc")
    } else {
      setSortColumn(column)
      setSortDirection("asc")
    }
    setCurrentPage(1)
  }

  const getClassificationBadge = (classification: string) => {
    switch (classification) {
      case "exoplanet":
        return <Badge className="bg-chart-4 text-white border-0">Exoplanet</Badge>
      case "not_exoplanet":
        return <Badge className="bg-chart-2 text-white border-0">Not Exoplanet</Badge>
      case "unsure":
        return <Badge className="bg-chart-5 text-white border-0">Unsure</Badge>
      default:
        return (
          <Badge variant="outline" className="border-muted-foreground/30 text-muted-foreground">
            Unclassified
          </Badge>
        )
    }
  }

  return (
    <Card className="p-6 bg-card border-2 border-primary/30 shadow-lg">
      <div className="flex flex-col md:flex-row gap-4 mb-6">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search candidates..."
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value)
              setCurrentPage(1)
            }}
            className="pl-10 border-2 border-primary/20 focus:border-primary bg-background text-foreground"
          />
        </div>
        <Select
          value={filterType}
          onValueChange={(v) => {
            setFilterType(v)
            setCurrentPage(1)
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
        <div className="max-h-[600px] overflow-y-auto">
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
                <th className="text-left p-3 text-sm font-semibold">Probability</th>
                <th className="text-left p-3 text-sm font-semibold">Classification</th>
                <th className="text-left p-3 text-sm font-semibold">Actions</th>
              </tr>
            </thead>
            <tbody className="bg-card">
              {paginatedData.length === 0 ? (
                <tr>
                  <td colSpan={columns.length + 4} className="text-center p-8 text-muted-foreground">
                    No candidates found
                  </td>
                </tr>
              ) : (
                <TooltipProvider>
                  {paginatedData.map((row, idx) => (
                    <motion.tr
                      key={row._index}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: idx * 0.01 }}
                      className="border-b border-border hover:bg-muted/50 transition-colors"
                    >
                      <td className="p-3 text-sm text-muted-foreground font-semibold">{row._index + 1}</td>
                      {columns.map((col) => (
                        <Tooltip key={col}>
                          <TooltipTrigger asChild>
                            <td className="p-3 text-sm text-foreground font-mono cursor-help">
                              {typeof row[col] === "number" ? row[col].toFixed(4) : String(row[col])}
                            </td>
                          </TooltipTrigger>
                          <TooltipContent side="top" className="bg-primary text-primary-foreground border-primary/50">
                            <p className="font-semibold">{col}</p>
                            <p className="text-xs opacity-90">
                              Value: {typeof row[col] === "number" ? row[col].toFixed(6) : String(row[col])}
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      ))}
                      <td className="p-3 text-sm">
                        <Badge variant="outline" className="font-mono border-primary/30 text-foreground">
                          {(row._probability * 100).toFixed(1)}%
                        </Badge>
                      </td>
                      <td className="p-3 text-sm">{getClassificationBadge(row._classification)}</td>
                      <td className="p-3">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => onSelectCandidate(row._index)}
                          className="hover:bg-primary/10 hover:text-primary"
                        >
                          <Eye className="w-4 h-4" />
                        </Button>
                      </td>
                    </motion.tr>
                  ))}
                </TooltipProvider>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="mt-4 flex items-center justify-between">
        <div className="text-sm text-muted-foreground">
          Showing {paginatedData.length > 0 ? (currentPage - 1) * ITEMS_PER_PAGE + 1 : 0} to{" "}
          {Math.min(currentPage * ITEMS_PER_PAGE, filteredData.length)} of {filteredData.length} candidates
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            disabled={currentPage === 1}
            className="border-primary/30"
          >
            <ChevronLeft className="w-4 h-4" />
          </Button>
          <span className="text-sm text-foreground font-medium">
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
        </div>
      </div>
    </Card>
  )
}
