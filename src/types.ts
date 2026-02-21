export interface PredictionResult {
  word: string;
  confidence: number;
}

export interface RKAYResponse {
  detected_sentence: string;
  confidence: number;
  engine: string;
}

export interface CalibrationExample {
  phrase_text: string;
  embedding_description: string;
}

export interface PatientContext {
  frequentPhrases: string[];
  lastMessages: string[];
  calibrationExamples: CalibrationExample[];
}
