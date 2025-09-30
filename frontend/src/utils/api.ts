export async function fetchWithFallback(path: string, opts?: RequestInit) {
  // When running in Docker (not on localhost), use the proxy path directly
  if (!window.location.host.includes('localhost:3000')) {
    // Use the proxy path when not running on localhost:3000
    const proxyPath = path;
    console.log(`Trying proxy path: ${proxyPath}`);
    try {
      const res = await fetch(proxyPath, opts);
      console.log(`Response status: ${res.status}`);
      if (!res.ok) throw new Error(`status:${res.status}`);
      const json = await res.json();
      console.log(`Response JSON:`, json);
      return json;
    } catch (e) {
      console.error(`Failed to fetch from proxy path ${proxyPath}:`, e);
      throw e;
    }
  } else {
    // When running on localhost:3000, use direct paths for development
    const candidates = [
      `http://localhost:8080${path}`,
      `http://localhost:8081${path}`,
      path,
    ];

    let lastErr: any = null;
    for (const url of candidates) {
      try {
        console.log(`Trying URL: ${url}`);
        const res = await fetch(url, opts);
        console.log(`Response status: ${res.status}`);
        if (!res.ok) throw new Error(`status:${res.status}`);
        const json = await res.json();
        console.log(`Response JSON:`, json);
        return json;
      } catch (e) {
        console.error(`Failed to fetch from ${url}:`, e);
        lastErr = e;
        // try next
      }
    }
    throw lastErr;
  }
}