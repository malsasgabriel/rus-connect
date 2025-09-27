export async function fetchWithFallback(path: string, opts?: RequestInit) {
  const candidates = [
    path,
    `http://localhost:8081${path}`,
    `http://localhost:8080${path}`,
  ];

  let lastErr: any = null;
  for (const url of candidates) {
    try {
      const res = await fetch(url, opts);
      if (!res.ok) throw new Error(`status:${res.status}`);
      return res.json();
    } catch (e) {
      lastErr = e;
      // try next
    }
  }
  throw lastErr;
}
