export class ApiError extends Error {
  status: number;
  payload: unknown;

  constructor(message: string, status: number, payload: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

function resolveBaseUrl(): string {
  const configured = import.meta.env.VITE_API_URL?.trim();
  if (configured) {
    return configured.replace(/\/$/, "");
  }
  // Return empty string to use relative URLs (proxied by Vite in dev)
  return "";
}

const BASE_URL = resolveBaseUrl();

function buildUrl(path: string): string {
  if (!path.startsWith("/")) {
    throw new Error(`API path must start with '/' but got '${path}'`);
  }
  // If BASE_URL is empty, use relative path (will be proxied by Vite dev server)
  if (BASE_URL === "") {
    return path;
  }
  return `${BASE_URL}${path}`;
}

export async function fetchWithFallback<T = any>(path: string, opts?: RequestInit): Promise<T> {
  const url = buildUrl(path);
  const response = await fetch(url, {
    ...opts,
    headers: {
      "Content-Type": "application/json",
      ...(opts?.headers ?? {}),
    },
  });

  const contentType = response.headers.get("content-type") ?? "";
  const isJSON = contentType.includes("application/json");
  const payload = isJSON ? await response.json() : await response.text();

  if (!response.ok) {
    throw new ApiError(`Request failed: ${response.status}`, response.status, payload);
  }

  return payload as T;
}
