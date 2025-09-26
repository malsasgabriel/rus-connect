/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_GATEWAY_WS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
