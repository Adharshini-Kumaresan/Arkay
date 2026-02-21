import { PatientContext } from './types';

export const CALIBRATED_SENTENCES = [
  "I am in pain.",
  "I feel cold.",
  "I am scared.",
  "Please turn me.",
  "I need suction.",
  "I want to sleep.",
  "Please adjust my pillow.",
  "I need the doctor.",
  "Please call the nurse.",
  "I need help immediately.",
  "I cannot breathe.",
  "I need to use the bathroom."
];

export const DEFAULT_PATIENT_CONTEXT: PatientContext = {
  frequentPhrases: [
    "Emergency",
    "Comfort",
    "Medical",
    "Response"
  ],
  lastMessages: [
    "Response"
  ],
  calibrationExamples: [
    {
      phrase_text: "Emergency",
      embedding_description: "Wide vertical opening on 'E', sustained aperture for 'mergency'."
    },
    {
      phrase_text: "Comfort",
      embedding_description: "Rounded lip protrusion for 'Co', quick closure for 'm', followed by 'fort'."
    }
  ]
};
