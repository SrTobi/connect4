export interface ModelInfo {
  id: string;
  label: string;
  architecture: string;
  url: string;
}
export interface PlayerConfig {
  ai: string;
  attempts: number;
  depth: number;
}
export interface Position {
  cells: number[];
  turn: number;
  result: number;
  ply: number;
}
export interface ColumnStat {
  column: number;
  legal: boolean;
  score: number;
  wins: number;
  draws: number;
  losses: number;
  eligible: boolean;
  winning: boolean;
}
export interface Analysis {
  side: number;
  ai: string;
  label: string;
  attempts: number;
  columns: ColumnStat[];
  chosen: number;
  elapsed: number;
  move: number;
}
export interface MoveResult {
  position: Position;
  column: number;
  row: number;
  side: number;
  analysis?: Analysis;
}
export interface Request {
  id: number;
  type: "init" | "reset" | "human" | "ai";
  column?: number;
  config?: PlayerConfig;
  models?: ModelInfo[];
}
export interface Response {
  id: number;
  result?: Position | MoveResult;
  error?: string;
  progress?: number;
}
