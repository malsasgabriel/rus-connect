package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestGatewayHealthzEndpoint(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.GET("/healthz", handleHealthz)

	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
}

func TestGatewayReadyzEndpointWithoutDependencies(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("ANALYTICS_ENGINE_URL", "http://127.0.0.1:59999")
	r := gin.New()
	r.GET("/readyz", handleReadyz)

	req := httptest.NewRequest(http.MethodGet, "/readyz", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 without deps, got %d", w.Code)
	}
}

func TestGatewayPortFromEnv(t *testing.T) {
	old := os.Getenv("API_GATEWAY_PORT")
	t.Cleanup(func() { _ = os.Setenv("API_GATEWAY_PORT", old) })
	_ = os.Setenv("API_GATEWAY_PORT", "18080")
	if os.Getenv("API_GATEWAY_PORT") != "18080" {
		t.Fatal("expected API_GATEWAY_PORT to be set")
	}
}
