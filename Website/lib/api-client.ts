/**
 * API Client for STAR AI Python Backend
 * Handles communication with the FastAPI server
 */

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

// Maximum number of items to send per batch to avoid large payloads
const MAX_BATCH_SIZE = 100;

export interface PredictionRequest {
  features?: number[] | number[][];
  data?: ExoplanetData[];
  user_id?: string;
}

export interface ExoplanetData {
  name?: string;
  OrbitalPeriod: number;
  OPup: number;
  OPdown: number;
  TransEpoch: number;
  TEup: number;
  TEdown: number;
  Impact: number;
  ImpactUp: number;
  ImpactDown: number;
  TransitDur: number;
  DurUp: number;
  DurDown: number;
  TransitDepth: number;
  DepthUp: number;
  DepthDown: number;
  PlanetRadius: number;
  RadiusUp: number;
  RadiusDown: number;
  EquilibriumTemp: number;
  InsolationFlux: number;
  InsolationUp: number;
  InsolationDown: number;
  TransitSNR: number;
  StellarEffTemp: number;
  SteffUp: number;
  SteffDown: number;
  StellarLogG: number;
  LogGUp: number;
  LogGDown: number;
  StellarRadius: number;
  SradUp: number;
  SradDown: number;
  RA: number;
  Dec: number;
  KeplerMag: number;
}

export interface PredictionResult {
  name: string;
  score: number;
  label: boolean;
}

export interface PredictionResponse {
  data: PredictionResult[];
}

export interface ModelInfo {
  model_name: string;
  input_features: number;
  filtered_to?: number;
  removed_columns?: string[];
  output_classes: string[];
  description: string;
}

/**
 * Convert CSV row data to ExoplanetData format
 * Handles mapping from various CSV column names to API format
 */
export function csvRowToExoplanetData(row: any, index: number): ExoplanetData {
  const getValue = (key: string): number => {
    const value = row[key];
    if (value === null || value === undefined || value === "") return 0.0;
    const num = parseFloat(value);
    return isNaN(num) ? 0.0 : num;
  };

  return {
    name:
      row.name || row.kepoi_name || row.kepler_name || `Candidate_${index + 1}`,
    OrbitalPeriod: getValue("koi_period") || getValue("OrbitalPeriod") || 0,
    OPup: getValue("koi_period_err1") || getValue("OPup") || 0,
    OPdown: getValue("koi_period_err2") || getValue("OPdown") || 0,
    TransEpoch: getValue("koi_time0bk") || getValue("TransEpoch") || 0,
    TEup: getValue("koi_time0bk_err1") || getValue("TEup") || 0,
    TEdown: getValue("koi_time0bk_err2") || getValue("TEdown") || 0,
    Impact: getValue("koi_impact") || getValue("Impact") || 0,
    ImpactUp: getValue("koi_impact_err1") || getValue("ImpactUp") || 0,
    ImpactDown: getValue("koi_impact_err2") || getValue("ImpactDown") || 0,
    TransitDur: getValue("koi_duration") || getValue("TransitDur") || 0,
    DurUp: getValue("koi_duration_err1") || getValue("DurUp") || 0,
    DurDown: getValue("koi_duration_err2") || getValue("DurDown") || 0,
    TransitDepth: getValue("koi_depth") || getValue("TransitDepth") || 0,
    DepthUp: getValue("koi_depth_err1") || getValue("DepthUp") || 0,
    DepthDown: getValue("koi_depth_err2") || getValue("DepthDown") || 0,
    PlanetRadius: getValue("koi_prad") || getValue("PlanetRadius") || 0,
    RadiusUp: getValue("koi_prad_err1") || getValue("RadiusUp") || 0,
    RadiusDown: getValue("koi_prad_err2") || getValue("RadiusDown") || 0,
    EquilibriumTemp: getValue("koi_teq") || getValue("EquilibriumTemp") || 0,
    InsolationFlux: getValue("koi_insol") || getValue("InsolationFlux") || 0,
    InsolationUp: getValue("koi_insol_err1") || getValue("InsolationUp") || 0,
    InsolationDown:
      getValue("koi_insol_err2") || getValue("InsolationDown") || 0,
    TransitSNR: getValue("koi_model_snr") || getValue("TransitSNR") || 0,
    StellarEffTemp: getValue("koi_steff") || getValue("StellarEffTemp") || 0,
    SteffUp: getValue("koi_steff_err1") || getValue("SteffUp") || 0,
    SteffDown: getValue("koi_steff_err2") || getValue("SteffDown") || 0,
    StellarLogG: getValue("koi_slogg") || getValue("StellarLogG") || 0,
    LogGUp: getValue("koi_slogg_err1") || getValue("LogGUp") || 0,
    LogGDown: getValue("koi_slogg_err2") || getValue("LogGDown") || 0,
    StellarRadius: getValue("koi_srad") || getValue("StellarRadius") || 0,
    SradUp: getValue("koi_srad_err1") || getValue("SradUp") || 0,
    SradDown: getValue("koi_srad_err2") || getValue("SradDown") || 0,
    RA: getValue("ra") || getValue("RA") || 0,
    Dec: getValue("dec") || getValue("Dec") || 0,
    KeplerMag: getValue("koi_kepmag") || getValue("KeplerMag") || 0,
  };
}

/**
 * Make a prediction for single or multiple exoplanet candidates
 * Automatically handles batching for large datasets
 */
export async function predict(
  data: ExoplanetData[] | number[] | number[][],
  userId: string = "web_client"
): Promise<PredictionResult[]> {
  try {
    // Check if it's array of ExoplanetData objects
    if (
      Array.isArray(data) &&
      data.length > 0 &&
      typeof data[0] === "object" &&
      "OrbitalPeriod" in data[0]
    ) {
      const exoplanetData = data as ExoplanetData[];

      // Handle large datasets by batching
      if (exoplanetData.length > MAX_BATCH_SIZE) {
        console.log(
          `[API] Processing ${exoplanetData.length} items in batches of ${MAX_BATCH_SIZE}`
        );
        return await predictInBatches(exoplanetData, userId);
      }

      // Single batch - direct API call
      return await predictSingleBatch(exoplanetData, userId);
    }

    // Handle raw feature arrays
    else {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          features: data,
          user_id: userId,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API Error (${response.status}): ${errorText}`);
      }

      const result: PredictionResponse = await response.json();
      return result.data;
    }
  } catch (error) {
    console.error("[API] Prediction error:", error);
    throw error;
  }
}

/**
 * Make a single batch prediction call to the API
 */
async function predictSingleBatch(
  data: ExoplanetData[],
  userId: string
): Promise<PredictionResult[]> {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      data: data,
      user_id: userId,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API Error (${response.status}): ${errorText}`);
  }

  const result: PredictionResponse = await response.json();
  return result.data;
}

/**
 * Process large datasets in batches to avoid payload size limits
 */
async function predictInBatches(
  data: ExoplanetData[],
  userId: string
): Promise<PredictionResult[]> {
  const results: PredictionResult[] = [];
  const totalBatches = Math.ceil(data.length / MAX_BATCH_SIZE);

  for (let i = 0; i < data.length; i += MAX_BATCH_SIZE) {
    const batch = data.slice(i, i + MAX_BATCH_SIZE);
    const batchNumber = Math.floor(i / MAX_BATCH_SIZE) + 1;

    console.log(
      `[API] Processing batch ${batchNumber}/${totalBatches} (${batch.length} items)`
    );

    try {
      // Call predictSingleBatch directly to avoid recursion
      const batchResults = await predictSingleBatch(batch, userId);
      results.push(...batchResults);
    } catch (error) {
      console.error(`[API] Error in batch ${batchNumber}:`, error);
      // Add placeholder results for failed batch
      const placeholders = batch.map((item, idx) => ({
        name: item.name || `Candidate_${i + idx + 1}`,
        score: 0.5,
        label: false,
      }));
      results.push(...placeholders);
    }
  }

  return results;
}

/**
 * Get model information
 */
export async function getModelInfo(): Promise<ModelInfo> {
  try {
    const response = await fetch(`${API_BASE_URL}/model-info`);

    if (!response.ok) {
      throw new Error(`API Error (${response.status})`);
    }

    return await response.json();
  } catch (error) {
    console.error("[API] Get model info error:", error);
    throw error;
  }
}

/**
 * Download the AI model
 */
export async function downloadModel(modelId: string = "all"): Promise<Blob> {
  try {
    const response = await fetch(
      `${API_BASE_URL}/export_model?model_id=${modelId}`
    );

    if (!response.ok) {
      throw new Error(`API Error (${response.status})`);
    }

    return await response.blob();
  } catch (error) {
    console.error("[API] Download model error:", error);
    throw error;
  }
}

/**
 * Check if API is available
 */
export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/models`, {
      method: "GET",
      signal: AbortSignal.timeout(5000), // 5 second timeout
    });
    return response.ok;
  } catch (error) {
    console.error("[API] Health check failed:", error);
    return false;
  }
}
