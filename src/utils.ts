/**
 * Utility functions for lip movement calculations
 */

export interface Point {
  x: number;
  y: number;
  z: number;
}

export type LipState = 'Closed' | 'Slightly Open' | 'Fully Open' | 'Speaking';

export const calculateDistance = (p1: Point, p2: Point): number => {
  return Math.sqrt(
    Math.pow(p1.x - p2.x, 2) + 
    Math.pow(p1.y - p2.y, 2) + 
    Math.pow(p1.z - p2.z, 2)
  );
};

export const getLipState = (
  innerDist: number, 
  outerDist: number, 
  velocity: number
): LipState => {
  if (velocity > 0.008) return 'Speaking';
  if (innerDist < 0.01) return 'Closed';
  if (innerDist < 0.04) return 'Slightly Open';
  return 'Fully Open';
};

// MediaPipe FaceMesh Indices for Lips
export const LIP_INDICES = {
  innerUpper: 13,
  innerLower: 14,
  outerUpper: 0,
  outerLower: 17,
  leftCorner: 61,
  rightCorner: 291,
  faceTop: 10,
  faceBottom: 152
};
