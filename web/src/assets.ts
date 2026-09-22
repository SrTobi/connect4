// Vite's base includes the trailing slash. Use it for runtime fetches too:
// workers resolve relative URLs against /assets/, not the document URL.
export function assetUrl(path: string): string {
  return import.meta.env.BASE_URL + path.replace(/^\/+/, "");
}
