"use client";

import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  AlertTriangle,
  CheckCircle2,
  XCircle,
  ArrowRight,
  Info,
} from "lucide-react";

interface ColumnMapperProps {
  csvColumns: string[];
  onConfirm: (mapping: Record<string, string>) => void;
  onCancel: () => void;
}

const REQUIRED_COLUMNS = [
  {
    key: "Confirmation",
    original: "koi_disposition",
    description: "Confirmation status of the candidate",
  },
  {
    key: "OrbitalPeriod",
    original: "koi_period",
    description: "Orbital period in days",
  },
  {
    key: "OPup",
    original: "koi_period_err1",
    description: "Orbital period upper uncertainty",
  },
  {
    key: "OPdown",
    original: "koi_period_err2",
    description: "Orbital period lower uncertainty",
  },
  { key: "TransEpoch", original: "koi_time0bk", description: "Transit epoch" },
  {
    key: "TEup",
    original: "koi_time0bk_err1",
    description: "Transit epoch upper uncertainty",
  },
  {
    key: "TEdown",
    original: "koi_time0bk_err2",
    description: "Transit epoch lower uncertainty",
  },
  { key: "Impact", original: "koi_impact", description: "Impact parameter" },
  {
    key: "ImpactUp",
    original: "koi_impact_err1",
    description: "Impact parameter upper uncertainty",
  },
  {
    key: "ImpactDown",
    original: "koi_impact_err2",
    description: "Impact parameter lower uncertainty",
  },
  {
    key: "TransitDur",
    original: "koi_duration",
    description: "Transit duration in hours",
  },
  {
    key: "DurUp",
    original: "koi_duration_err1",
    description: "Transit duration upper uncertainty",
  },
  {
    key: "DurDown",
    original: "koi_duration_err2",
    description: "Transit duration lower uncertainty",
  },
  {
    key: "TransitDepth",
    original: "koi_depth",
    description: "Transit depth in parts per million",
  },
  {
    key: "DepthUp",
    original: "koi_depth_err1",
    description: "Transit depth upper uncertainty",
  },
  {
    key: "DepthDown",
    original: "koi_depth_err2",
    description: "Transit depth lower uncertainty",
  },
  {
    key: "PlanetRadius",
    original: "koi_prad",
    description: "Planet radius in Earth radii",
  },
  {
    key: "RadiusUp",
    original: "koi_prad_err1",
    description: "Planet radius upper uncertainty",
  },
  {
    key: "RadiusDown",
    original: "koi_prad_err2",
    description: "Planet radius lower uncertainty",
  },
  {
    key: "EquilibriumTemp",
    original: "koi_teq",
    description: "Equilibrium temperature in Kelvin",
  },
  {
    key: "TempUp",
    original: "koi_teq_err1",
    description: "Equilibrium temp upper uncertainty",
  },
  {
    key: "TempDown",
    original: "koi_teq_err2",
    description: "Equilibrium temp lower uncertainty",
  },
  {
    key: "InsolationFlux",
    original: "koi_insol",
    description: "Insolation flux in Earth flux",
  },
  {
    key: "InsolationUp",
    original: "koi_insol_err1",
    description: "Insolation flux upper uncertainty",
  },
  {
    key: "InsolationDown",
    original: "koi_insol_err2",
    description: "Insolation flux lower uncertainty",
  },
  {
    key: "TransitSNR",
    original: "koi_model_snr",
    description: "Transit signal-to-noise ratio",
  },
  {
    key: "StellarEffTemp",
    original: "koi_steff",
    description: "Stellar effective temperature in Kelvin",
  },
  {
    key: "SteffUp",
    original: "koi_steff_err1",
    description: "Stellar temp upper uncertainty",
  },
  {
    key: "SteffDown",
    original: "koi_steff_err2",
    description: "Stellar temp lower uncertainty",
  },
  {
    key: "StellarLogG",
    original: "koi_slogg",
    description: "Stellar surface gravity (log10)",
  },
  {
    key: "LogGUp",
    original: "koi_slogg_err1",
    description: "Stellar log g upper uncertainty",
  },
  {
    key: "LogGDown",
    original: "koi_slogg_err2",
    description: "Stellar log g lower uncertainty",
  },
  {
    key: "StellarRadius",
    original: "koi_srad",
    description: "Stellar radius in solar radii",
  },
  {
    key: "SradUp",
    original: "koi_srad_err1",
    description: "Stellar radius upper uncertainty",
  },
  {
    key: "SradDown",
    original: "koi_srad_err2",
    description: "Stellar radius lower uncertainty",
  },
  {
    key: "RA",
    original: "ra",
    description: "Right ascension in decimal degrees",
  },
  {
    key: "Dec",
    original: "dec",
    description: "Declination in decimal degrees",
  },
  { key: "KeplerMag", original: "koi_kepmag", description: "Kepler magnitude" },
];

export function ColumnMapper({
  csvColumns,
  onConfirm,
  onCancel,
}: ColumnMapperProps) {
  const [mapping, setMapping] = useState<Record<string, string>>({});
  const [showWarning, setShowWarning] = useState(false);

  // Auto-detect column mappings with improved algorithm
  useEffect(() => {
    const autoMapping: Record<string, string> = {};

    // Database mappings: [K2_value, K1/Kepler_value]
    // K2 format: disposition, pl_orbper, pl_rade, st_teff, st_rad, etc.
    // K1 format: koi_disposition, koi_period, koi_prad, koi_steff, koi_srad, etc.
    const databaseMappings: Record<string, [string | null, string | null]> = {
      Confirmation: ["disposition", "koi_disposition"],
      OrbitalPeriod: ["pl_orbper", "koi_period"],
      OPup: ["pl_orbpererr1", "koi_period_err1"],
      OPdown: ["pl_orbpererr2", "koi_period_err2"],
      TransEpoch: [null, "koi_time0bk"],
      TEup: [null, "koi_time0bk_err1"],
      TEdown: [null, "koi_time0bk_err2"],
      Impact: [null, "koi_impact"],
      ImpactUp: [null, "koi_impact_err1"],
      ImpactDown: [null, "koi_impact_err2"],
      TransitDur: [null, "koi_duration"],
      DurUp: [null, "koi_duration_err1"],
      DurDown: [null, "koi_duration_err2"],
      TransitDepth: [null, "koi_depth"],
      DepthUp: [null, "koi_depth_err1"],
      DepthDown: [null, "koi_depth_err2"],
      PlanetRadius: ["pl_rade", "koi_prad"],
      RadiusUp: ["pl_radeerr1", "koi_prad_err1"],
      RadiusDown: ["pl_radeerr2", "koi_prad_err2"],
      EquilibriumTemp: ["pl_eqt", "koi_teq"],
      TempUp: ["pl_eqterr1", "koi_teq_err1"],
      TempDown: ["pl_eqterr2", "koi_teq_err2"],
      InsolationFlux: ["pl_insol", "koi_insol"],
      InsolationUp: ["pl_insolerr1", "koi_insol_err1"],
      InsolationDown: ["pl_insolerr2", "koi_insol_err2"],
      TransitSNR: [null, "koi_model_snr"],
      StellarEffTemp: ["st_teff", "koi_steff"],
      SteffUp: ["st_tefferr1", "koi_steff_err1"],
      SteffDown: ["st_tefferr2", "koi_steff_err2"],
      StellarLogG: [null, "koi_slogg"],
      LogGUp: [null, "koi_slogg_err1"],
      LogGDown: [null, "koi_slogg_err2"],
      StellarRadius: ["st_rad", "koi_srad"],
      SradUp: ["st_raderr1", "koi_srad_err1"],
      SradDown: ["st_raderr2", "koi_srad_err2"],
      RA: ["ra", "ra"],
      Dec: ["dec", "dec"],
      KeplerMag: [null, "koi_kepmag"],
    };

    // STEP 1: Detect which database format is being used
    let k2Score = 0;
    let k1Score = 0;

    const csvColumnsLower = csvColumns.map((col) => col.toLowerCase());

    // Count matches for each database format
    Object.values(databaseMappings).forEach(([k2Col, k1Col]) => {
      if (k2Col && csvColumnsLower.includes(k2Col.toLowerCase())) {
        k2Score++;
      }
      if (k1Col && csvColumnsLower.includes(k1Col.toLowerCase())) {
        k1Score++;
      }
    });

    // Determine which database format to use (0 = K2, 1 = K1, null = neither/use fuzzy)
    let detectedDB: 0 | 1 | null = null;
    if (k2Score > k1Score && k2Score >= 3) {
      detectedDB = 0; // K2 format detected
    } else if (k1Score > k2Score && k1Score >= 3) {
      detectedDB = 1; // K1/Kepler format detected
    }

    // Short column names that should ONLY match exactly (prevent false positives)
    const strictMatchColumns = new Set(["ra", "dec"]);

    // Normalize function for better matching
    const normalize = (str: string) =>
      str
        .toLowerCase()
        .replace(/[_\s\-\.]/g, "") // Remove underscores, spaces, hyphens, dots
        .replace(/up$|down$|err1$|err2$/g, ""); // Remove common suffixes

    REQUIRED_COLUMNS.forEach((col) => {
      let bestMatch: string | null = null;
      let bestScore = 0;

      // STEP 2: If database format detected, use exact mappings from that DB
      if (detectedDB !== null && databaseMappings[col.key]) {
        const dbColumnName = databaseMappings[col.key][detectedDB];
        if (dbColumnName) {
          // Look for exact match in CSV columns
          const matchedCol = csvColumns.find(
            (csvCol) => csvCol.toLowerCase() === dbColumnName.toLowerCase()
          );
          if (matchedCol) {
            autoMapping[col.key] = matchedCol;
            return; // Skip fuzzy matching - we have exact DB match
          }
        }
        // If dbColumnName is null for this DB, fall through to fuzzy matching
      }

      // STEP 3: Fuzzy matching (used when no DB detected or column not in detected DB)
      csvColumns.forEach((csvCol) => {
        let score = 0;
        const csvNorm = normalize(csvCol);
        const keyNorm = normalize(col.key);
        const origNorm = normalize(col.original);
        const csvLower = csvCol.toLowerCase();
        const origLower = col.original.toLowerCase();

        // Special handling for strict match columns (ra, dec)
        if (strictMatchColumns.has(origLower)) {
          // Only allow exact matches for these columns
          if (csvLower === origLower) {
            score = 100;
          }
          // Skip any fuzzy matching for strict columns
        } else {
          // 1. Exact match - 100 points
          if (csvLower === origLower || csvLower === col.key.toLowerCase()) {
            score = 100;
          }
          // 2. Exact match after normalization - 90 points
          else if (csvNorm === origNorm || csvNorm === keyNorm) {
            score = 90;
          }
          // 3. Original name contains CSV column (only if CSV column is long enough) - 80 points
          else if (
            csvCol.length >= 4 &&
            (origLower.includes(csvLower) || csvLower.includes(origLower))
          ) {
            score = 80;
          }
          // 4. Normalized contains match (only if CSV is long enough) - 70 points
          else if (
            csvCol.length >= 4 &&
            (origNorm.includes(csvNorm) || csvNorm.includes(origNorm))
          ) {
            score = 70;
          }
          // 5. Key name similarity (only if CSV is long enough) - 60 points
          else if (
            csvCol.length >= 4 &&
            (keyNorm.includes(csvNorm) || csvNorm.includes(keyNorm))
          ) {
            score = 60;
          }
          // 6. Description keywords match - 50 points
          else {
            const descWords = col.description.toLowerCase().split(/\s+/);
            const csvWords = csvLower.split(/[_\s\-\.]/);
            const matchingWords = descWords.filter(
              (word) =>
                word.length > 3 &&
                csvWords.some(
                  (csvWord) =>
                    csvWord.length > 3 &&
                    (csvWord.includes(word) || word.includes(csvWord))
                )
            );
            if (matchingWords.length > 0) {
              score = 50 + matchingWords.length * 5;
            }
          }

          // Special handling for error columns (err1, err2, up, down)
          if (col.key.includes("Up") || col.key.includes("Down")) {
            if (csvLower.includes("err1") && col.key.includes("Up"))
              score += 10;
            if (csvLower.includes("err2") && col.key.includes("Down"))
              score += 10;
            if (csvLower.includes("up") && col.key.includes("Up")) score += 10;
            if (csvLower.includes("down") && col.key.includes("Down"))
              score += 10;
          }

          // Common aliases and variations (only for longer, more specific aliases)
          const aliases: Record<string, string[]> = {
            OrbitalPeriod: [
              "period",
              "orbital_period",
              "orb_period",
              "koi_period",
            ],
            PlanetRadius: ["radius", "planet_radius", "prad", "koi_prad"],
            TransitDepth: ["depth", "transit_depth", "koi_depth"],
            TransitDur: ["duration", "transit_duration", "koi_duration"],
            EquilibriumTemp: [
              "temperature",
              "temp",
              "teq",
              "koi_teq",
              "equilibrium",
            ],
            InsolationFlux: ["insolation", "flux", "insol", "koi_insol"],
            StellarEffTemp: ["stellar_temp", "steff", "koi_steff", "star_temp"],
            StellarRadius: [
              "stellar_radius",
              "srad",
              "koi_srad",
              "star_radius",
            ],
            StellarLogG: ["logg", "slogg", "koi_slogg", "surface_gravity"],
            Impact: ["impact", "impact_param", "koi_impact"],
            TransitSNR: [
              "snr",
              "signal_noise",
              "koi_model_snr",
              "signal_to_noise",
            ],
            KeplerMag: ["kepmag", "koi_kepmag", "magnitude"],
            RA: ["right_ascension", "rightascension"],
            Dec: ["declination"],
            Confirmation: [
              "disposition",
              "koi_disposition",
              "status",
              "confirmed",
            ],
            TransEpoch: ["epoch", "time0", "koi_time0bk", "transit_time"],
          };

          if (aliases[col.key]) {
            // Only match if the alias is substantially present (not just a substring)
            const matchedAlias = aliases[col.key].find(
              (alias) =>
                csvLower === alias ||
                (alias.length >= 4 && csvLower.includes(alias))
            );
            if (matchedAlias) {
              score += 15;
            }
          }
        }

        // Update best match if this score is higher
        if (score > bestScore) {
          bestScore = score;
          bestMatch = csvCol;
        }
      });

      // Only auto-map if confidence is high enough (score >= 50)
      if (bestMatch && bestScore >= 50) {
        autoMapping[col.key] = bestMatch;
      }
    });

    setMapping(autoMapping);
  }, [csvColumns]);

  const handleMappingChange = (standardKey: string, csvColumn: string) => {
    setMapping((prev) => {
      const newMapping = { ...prev };
      if (csvColumn === "none") {
        delete newMapping[standardKey];
      } else {
        newMapping[standardKey] = csvColumn;
      }
      return newMapping;
    });
  };

  const getMappedCount = () => Object.keys(mapping).length;
  const getTotalColumns = () => REQUIRED_COLUMNS.length;

  const getPrecisionLevel = () => {
    const mappedCount = getMappedCount();
    const totalCount = getTotalColumns();
    const percentage = (mappedCount / totalCount) * 100;

    if (percentage === 0)
      return {
        level: "none",
        message: "No columns mapped - Cannot proceed",
        color: "text-red-500",
      };
    if (percentage < 30)
      return {
        level: "very-low",
        message: "Very low precision - Most critical columns missing",
        color: "text-red-500",
      };
    if (percentage < 50)
      return {
        level: "low",
        message: "Low precision - Many important columns missing",
        color: "text-orange-500",
      };
    if (percentage < 70)
      return {
        level: "medium",
        message: "Medium precision - Some columns missing",
        color: "text-yellow-500",
      };
    if (percentage < 90)
      return {
        level: "good",
        message: "Good precision - Most columns mapped",
        color: "text-blue-500",
      };
    return {
      level: "excellent",
      message: "Excellent precision - All or nearly all columns mapped",
      color: "text-green-500",
    };
  };

  const handleConfirm = () => {
    const mappedCount = getMappedCount();

    if (mappedCount === 0) {
      setShowWarning(true);
      return;
    }

    const precision = getPrecisionLevel();
    if (
      precision.level === "none" ||
      precision.level === "very-low" ||
      precision.level === "low"
    ) {
      setShowWarning(true);
      return;
    }

    onConfirm(mapping);
  };

  const handleForceConfirm = () => {
    if (getMappedCount() === 0) return; // Never allow 0 columns
    onConfirm(mapping);
  };

  const precision = getPrecisionLevel();

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <Card className="w-full max-w-6xl max-h-[90vh] flex flex-col bg-card border-2 border-border shadow-2xl">
        {/* Header */}
        <div className="border-b border-border px-6 py-4 flex-shrink-0">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-2xl font-bold text-foreground">
              Map CSV Columns
            </h2>
            <Button
              variant="ghost"
              size="icon"
              onClick={onCancel}
              className="hover:bg-destructive/10 hover:text-destructive"
            >
              <XCircle className="w-5 h-5" />
            </Button>
          </div>
          <p className="text-sm text-muted-foreground mb-3">
            Match your CSV columns to the standardized format required by the AI
            model. All columns are optional but affect prediction accuracy.
          </p>

          {/* Precision Indicator */}
          <div className="flex items-center gap-3 bg-muted/50 rounded-lg p-3">
            <div className="flex-1">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-semibold text-foreground">
                  Mapping Progress: {getMappedCount()} / {getTotalColumns()}
                </span>
                <Badge className={`${precision.color} bg-transparent border`}>
                  {precision.level.toUpperCase()}
                </Badge>
              </div>
              <div className="w-full bg-muted rounded-full h-2 border border-border">
                <div
                  className={`h-2 rounded-full transition-all duration-300 ${
                    precision.level === "excellent"
                      ? "bg-green-500"
                      : precision.level === "good"
                      ? "bg-blue-500"
                      : precision.level === "medium"
                      ? "bg-yellow-500"
                      : "bg-red-500"
                  }`}
                  style={{
                    width: `${(getMappedCount() / getTotalColumns()) * 100}%`,
                  }}
                />
              </div>
            </div>
            <Tooltip>
              <TooltipTrigger>
                <Info className="w-5 h-5 text-muted-foreground" />
              </TooltipTrigger>
              <TooltipContent>
                <p className="max-w-xs">{precision.message}</p>
              </TooltipContent>
            </Tooltip>
          </div>
        </div>

        {/* Mapping Table */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          <div className="space-y-2">
            {REQUIRED_COLUMNS.map((col) => (
              <div
                key={col.key}
                className="flex items-center gap-4 p-3 bg-muted/30 rounded-lg border border-border hover:border-primary/50 transition-colors"
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <p className="font-semibold text-sm text-foreground">
                      {col.key}
                    </p>
                    {mapping[col.key] && (
                      <CheckCircle2 className="w-4 h-4 text-green-500 flex-shrink-0" />
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground truncate">
                    {col.description} • Original:{" "}
                    <code className="bg-muted px-1 py-0.5 rounded">
                      {col.original}
                    </code>
                  </p>
                </div>

                <ArrowRight className="w-4 h-4 text-muted-foreground flex-shrink-0" />

                <Select
                  value={mapping[col.key] || "none"}
                  onValueChange={(value) => handleMappingChange(col.key, value)}
                >
                  <SelectTrigger className="w-64 bg-background">
                    <SelectValue placeholder="Select CSV column..." />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">
                      <span className="text-muted-foreground">Not mapped</span>
                    </SelectItem>
                    {csvColumns.map((csvCol) => (
                      <SelectItem key={csvCol} value={csvCol}>
                        {csvCol}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-border px-6 py-4 flex-shrink-0">
          {showWarning && (
            <div
              className={`mb-4 p-4 rounded-lg border-2 ${
                getMappedCount() === 0
                  ? "bg-red-500/10 border-red-500"
                  : "bg-yellow-500/10 border-yellow-500"
              }`}
            >
              <div className="flex items-start gap-3">
                <AlertTriangle
                  className={`w-5 h-5 flex-shrink-0 ${
                    getMappedCount() === 0 ? "text-red-500" : "text-yellow-500"
                  }`}
                />
                <div className="flex-1">
                  <p
                    className={`font-semibold mb-1 ${
                      getMappedCount() === 0
                        ? "text-red-500"
                        : "text-yellow-500"
                    }`}
                  >
                    {getMappedCount() === 0
                      ? "Cannot Proceed"
                      : "Warning: Low Precision"}
                  </p>
                  <p className="text-sm text-foreground/80">
                    {getMappedCount() === 0
                      ? "You must map at least one column to proceed. The AI model requires data to make predictions."
                      : precision.message +
                        ". The AI predictions may not be accurate with this few columns mapped. Consider mapping more columns for better results."}
                  </p>
                  {getMappedCount() > 0 && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleForceConfirm}
                      className="mt-3 border-yellow-500 text-yellow-600 hover:bg-yellow-500/10"
                    >
                      Proceed Anyway
                    </Button>
                  )}
                </div>
              </div>
            </div>
          )}

          <div className="flex items-center justify-between">
            <Button variant="outline" onClick={onCancel}>
              Cancel
            </Button>
            <Button
              onClick={handleConfirm}
              disabled={getMappedCount() === 0}
              className="bg-primary hover:bg-primary/90"
            >
              Confirm Mapping
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
}
